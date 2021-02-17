#include "dms_base.h"
#include <petscviewerhdf5.h>
#include <sstream>
#include <cstdio>
#include <cstdlib>

#define LARGE_NUMBER 1000

int DmsBase::getNumSS() {
        return numSS;
}

PetscErrorCode readRef(char* fname, DmsBase& Dbase) {
	PetscFunctionBegin;

	Dbase.fpLog << Dbase.getTime() << ":INFO:Reading reference configuration from " << fname << std::endl;
	PetscErrorCode ierr;

	try {
        	FILE * fp;
        	fp = fopen (fname, "r");
        	double coords[Dbase.Microscopic->Get_Dim()];

        	char buffer[LARGE_NUMBER];
        	fgets(buffer, LARGE_NUMBER, fp);
        	fgets(buffer, LARGE_NUMBER, fp);

        	std::string numAtomsStr = buffer;
        	std::string::size_type sz;
       	 	int numAtoms = std::stoi (numAtomsStr, &sz);

        	int resNum, atomNum;
        	char resID[LARGE_NUMBER], atomID[LARGE_NUMBER];

        	for(int i = 0; i < numAtoms; i++) {

                	fscanf(fp, "%5d%5s%5s%5d", &resNum, resID, atomID, &atomNum);

                	if( strncmp(resID, "SOL", 3) && strncmp(resID, "NA", 2) && strncmp(resID, "CL", 2) && strncmp(resID, "GRA", 3) ) { // this has to take into account other ions of non-protein atoms
                        	fscanf(fp, "%lf%lf%lf", &coords[0], &coords[1], &coords[2]);

				for(int dim = 0; dim < Dbase.Microscopic->Get_Dim(); dim++) {

					ierr = VecSetValues(Dbase.Microscopic->Get_RefCoords()[dim], 1, &i, &coords[dim], INSERT_VALUES);
                        		CHKERRQ(ierr);
				}
                	}
        	}

        	fclose(fp);
	}
	catch (int e) {
                        std::cout << "An exception has occured in reading ref config file. Exception Nr. " << e << std::endl;
        }
	
	PetscFunctionReturn(ierr);
}

PetscViewer openPetsc() {
        PetscFunctionBegin;
        PetscViewer    viewer;

	std::ifstream ifile("dms.h5");

	if(ifile)
        	PetscViewerHDF5Open(PETSC_COMM_SELF, "dms.h5", FILE_MODE_APPEND, &viewer);
	else
		PetscViewerHDF5Open(PETSC_COMM_SELF, "dms.h5", FILE_MODE_WRITE, &viewer);

	ifile.close();

	PetscViewerHDF5PushGroup(viewer, "/");

	PetscFunctionReturn(viewer);
}

PetscErrorCode closePetsc(PetscViewer* viewer) {
	PetscFunctionBegin;
        PetscErrorCode ierr = PetscViewerDestroy(viewer);

	CHKERRQ(ierr);

	PetscFunctionReturn(ierr);
}

PetscErrorCode writePetsc(CVec coords, std::vector <char*> keys, PetscInt step, PetscViewer* viewer) {
        PetscFunctionBegin;

	PetscErrorCode ierr;
	std::ostringstream oss;
	oss << step;
 
	std::string pathToTime = "/" + oss.str();
	ierr = PetscViewerHDF5PushGroup(*viewer, pathToTime.c_str());
	CHKERRQ(ierr);

        for(auto i = 0; i < coords.size(); i++) {

                ierr = VecView(coords[i], *viewer);
                CHKERRQ(ierr);
        }

        //PetscViewerHDF5SetTimestep(*viewer, step); // what does this do?!
	PetscViewerFlush(*viewer);

        PetscFunctionReturn(ierr);
}

/*
std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);

    return elems;
}
*/

// Move this function somewhere else
gmx_bool dmsInitialize(int argc, char* argv[]) {
	PetscFunctionBegin;

	PetscInitialize(&argc, &argv, NULL, NULL);

	return 0;
}

void DmsBase::registerMethods() {
	PetscFunctionBegin;

	std::string SpaceWarping = "SpaceWarping",
        FieldVariables = "FieldVariables";

    	fineGrainHash[SpaceWarping] = swm::fineGrain;
    	coarseGrainHash[SpaceWarping] = swm::coarseGrain;
	coarseGrainVeloHash[SpaceWarping] = swm::coarseGrainVelo;
    	initializeHash[SpaceWarping] = swm::initialize;
    	updateRefHash[SpaceWarping] = swm::updateRef;

    	fineGrainHash[FieldVariables] = swm::fineGrain;
    	coarseGrainHash[FieldVariables] = swm::coarseGrain;
    	initializeHash[FieldVariables] = swm::initialize;
    	updateRefHash[FieldVariables] = swm::updateRef;
}

DmsBase::DmsBase(const t_state* state, const t_mdatoms* tmdatoms,
		const gmx_mtop_t* top, const t_inputrec* ir, const gmx_int64_t aDim, const gmx_int64_t cDim, const gmx_int64_t nCGx,
		const gmx_int64_t nCGy, const gmx_int64_t nCGz, const gmx_int64_t freq, const gmx_int64_t assFreq, const real dt, 
		MPI_Comm communic, const int mSteps, const real resol, const real scaling, const gmx_int64_t thresh, const int nHist, 
		const PetscInt ssIndex, const PetscInt nSS, std::string cgType, bool deb, char* topFname, char* selFname) : freqUpdate(freq), 
		timeStep(0), dim(aDim), cgDim(cDim), assembleFreq(assFreq), debug(deb) {

	PetscFunctionBegin;

	comm = communic;

	MPI_Comm_size(PETSC_COMM_WORLD, &mpiSize);
	MPI_Comm_rank(PETSC_COMM_WORLD, &mpiRank);

	tprRecord = ir;
	cgMethod = cgType;
	nHistory = nHist;
	numSS = nSS;
	Delta = dt;

	if(nHist > 1)
		Integrator = new DmsPade(dt, mSteps / 1000.0);
	else
		Integrator = new DmsEuler(dt);

	registerMethods();

    	if(!mpiRank) {

		fpLog.open("dms.log", std::ios::out);
		fpLog << getTime() << ":INFO:Initializing constructor" << std::endl;

    		// Make sure the cgMethod exists
    		if ( fineGrainHash.find(cgMethod) == fineGrainHash.end() ) {
            		fpLog << getTime() << "ERROR:CG method pass does not exist. Choose one of the following instead:" << std::endl;

            	        for( auto it = fineGrainHash.begin(); it != fineGrainHash.end(); ++it)
                    		fpLog << getTime() << it->first << std::endl;

            		throw std::invalid_argument("CG method has not yet been implemented");
    		}

           	Microscopic = new Micro_state(state, tmdatoms, top, ir, dim, comm, fineGrainHash[cgMethod], mSteps, dt, numSS, ssIndex, this, topFname, selFname);
           	nAtoms = Microscopic->Get_DOF();
		nLocalAtoms = Microscopic->Get_DOF_local();

		fpLog << getTime() << ":INFO:Found " << nAtoms << " atoms pertaining to the given macromolecule" << std::endl;

		ierr = initializeHash[cgMethod](*this);

            	fpLog << getTime() << ":INFO:Setting up " << cgMethod << " subsystem" << std::endl;

            	Mesoscopic  = new Meso_state(nCG, cgDim, nHist, comm, coarseGrainHash[cgMethod], coarseGrainVeloHash[cgMethod]);
            	fpLog << getTime() << ":INFO:Success! Number of CG variables per subsystem is " << nCG << std::endl;
    	}   
}

DmsBase::DmsBase(const t_state* state, const t_mdatoms* tmdatoms,
		const gmx_mtop_t* top, const t_inputrec* ir, const gmx_int64_t aDim, const gmx_int64_t cDim, const int maxOrder,
		const gmx_int64_t freq, const real dt, const gmx_int64_t t0, MPI_Comm communic, const int mSteps, const double optimScale, const PetscInt nHist, 
		const PetscInt ssIndex, const PetscInt nss, std::string cgType, char* userRef, char* topFname, char* selFname) { 

	PetscFunctionBegin;

	freqUpdate = freq;
	timeStep = t0;
	dim = aDim; 
	cgDim = cDim;
	order = maxOrder;
	comm = communic;
	scale = optimScale;
	numSS = nss;
	nHistory = nHist;
	Delta = dt;

	MPI_Comm_size(PETSC_COMM_WORLD, &mpiSize);
	MPI_Comm_rank(PETSC_COMM_WORLD, &mpiRank);

	tprRecord = ir;
	cgMethod = cgType;

	if(nHist > 1) {

		if(!mpiRank)		
			std::cout << "Using Pade integrator" << std::endl;

                Integrator = new DmsPade(dt, mSteps / 1000.0);
	}
        else {

		if(!mpiRank)
			std::cout << "Using Euler integrator" << std::endl;

            	Integrator = new DmsEuler(dt);
	}

	registerMethods();
	viewer = openPetsc();

	if(!mpiRank) {

		fpLog.open("dms.log", std::ios::out);
        	fpLog << getTime() << ":INFO:Initializing constructor" << std::endl;

		// Make sure the cgMethod exists
		if ( fineGrainHash.find(cgMethod) == fineGrainHash.end() ) {
                	fpLog << getTime() << ":ERROR:CG Method supplied does not exist. Choose one of the following instead:" << std::endl;

                	for( auto it = fineGrainHash.begin(); it != fineGrainHash.end(); ++it)
                        	fpLog << getTime() << it->first << std::endl;

                	throw std::invalid_argument("CG method has not yet been implemented");
        	}

               	Microscopic = new Micro_state(state, tmdatoms, top, ir, dim, comm, fineGrainHash[cgMethod], mSteps, dt, numSS, ssIndex, this, topFname, selFname);

		nAtoms = Microscopic->Get_DOF();
		nLocalAtoms = Microscopic->Get_DOF_local();

		ierr = initializeHash[cgMethod](*this);

		if(userRef)
			ierr = readRef(userRef, *this);

                fpLog << getTime() << ":INFO:Setting up " << cgMethod << " subsystem ..." << std::endl;

                Mesoscopic  = new Meso_state(nCG, cgDim, nHist, comm, coarseGrainHash[cgMethod], coarseGrainVeloHash[cgMethod]);
                fpLog << getTime() << ":INFO:Success! Number of CG variables per subsystem is " << nCG << std::endl;
		fpLog << getTime() << ":INFO:Reference structure will be updated every " << freqUpdate <<  " ps" << std::endl;
        }
}

int DmsBase::cgStep(gmx_int64_t gromacStep) {
	/*
	 * Whenever this function is called, the one thing that *must* be always done is gather the Coords
	 * Vec in order to initiate computations.
	 * If the ref structure is to be updated, then coarse-graining must be performed again.
	 *
	 * This function calls three methods:
	 * CoarseGraing -> dimensionality reduction
	 * Integrate    -> advance the CG state in time
	 * FineGraining -> recover the atomistic state
	 * Comments with '!!!' are used for adding new integrator. Remember to delete them.
	 */

	PetscFunctionBegin;

	// Do serial computations for SWM
	if(!mpiRank) {

		fpLog << getTime() << ":INFO:Taking CG time step " << timeStep << std::endl;
		timeStep++; // !!!cg step number, not total simulation step number
		CHKERRQ(ierr);


		// Pade: save CG state at t = - Delta
		
		for(int dim = 0; dim < Mesoscopic->Get_Dim(); dim++) {

			ierr = VecCopy(Mesoscopic->Get_Coords()[dim], Mesoscopic->getCGTensor()[0][dim]);

			for(int fr = Mesoscopic->getCGTensor().size() - 1; fr > 0; fr--) {
                        	ierr = VecCopy(Mesoscopic->getCGTensor()[fr-1][dim], Mesoscopic->getCGTensor()[fr][dim]);
                        	CHKERRQ(ierr);
                	}
		}
		

		// Coarse-grain before MD phase (initial CGs) should have already been called during halfMD stage

		for(int dim = 0; dim < Mesoscopic->Get_Dim(); dim++) {
                        ierr = VecCopy(Mesoscopic->Get_Coords()[dim], Mesoscopic->Get_pCoords()[dim]);
                        CHKERRQ(ierr);
                }

		for(int dim = 0; dim < Mesoscopic->Get_Dim(); dim++) {
                        ierr = VecCopy(Mesoscopic->Get_Velocities()[dim], Mesoscopic->Get_pVelocities()[dim]);
                        CHKERRQ(ierr);
                }

		if(timeStep == 1)
			for(int dim = 0; dim < Mesoscopic->Get_Dim(); dim++) {

                        	ierr = VecCopy(Mesoscopic->Get_Coords()[dim], Mesoscopic->Get_RefCoords()[dim]);
                        	CHKERRQ(ierr);		
                	} // !!!The first cg state is used as the reference state which will be updated

		// save current coords to be potentially used for fine-graining
		for(int dim = 0; dim < Microscopic->Get_Dim(); dim++) {
                        ierr = VecCopy(Microscopic->Get_Coords()[dim], Microscopic->Get_pCoords()[dim]);
                        CHKERRQ(ierr);
                }

		fpLog << getTime() << ":INFO:Syncing DMS with Gromacs" << std::endl;

		// This function syncs the coords, velocities, etc. with the GROMACS t_state
		ierr = Microscopic->Sync_DMS_fromMD(this);
		CHKERRQ(ierr);

		// update the ref config if needed based on the relaxed microstate
		ierr = updateRefHash[cgMethod](*this);
        	CHKERRQ(ierr);

		// Coarse-grain after MD phase(final Coords)
		ierr = constructCoords();
		CHKERRQ(ierr);

		for(int dim = 0; dim < Mesoscopic->Get_Dim(); dim++)
                        ierr = VecCopy(Mesoscopic->Get_Velocities()[dim], Mesoscopic->Get_pVelocities()[dim]);

		ierr = constructVelocities();
		CHKERRQ(ierr);
		
		ierr = constructForces();
                CHKERRQ(ierr);

		// Pade: save CG state at t = -	Delta + delta
		
                for(int dim = 0; dim < Mesoscopic->Get_Dim(); dim++) {

                        ierr = VecCopy(Mesoscopic->Get_Coords()[dim], Mesoscopic->getCGTensor()[1][dim]);

                        for(int fr = Mesoscopic->getCGTensor().size() - 1; fr > 1; fr--) {
                                ierr = VecCopy(Mesoscopic->getCGTensor()[fr-1][dim], Mesoscopic->getCGTensor()[fr][dim]);
                                CHKERRQ(ierr);
                       	}
                }
		

		for(int dim = 0; dim < Mesoscopic->Get_Dim(); dim++) {
			ierr = VecCopy(Mesoscopic->Get_Coords()[dim], Mesoscopic->Get_pCoords()[dim]);
			CHKERRQ(ierr);
		}

		fpLog << getTime() << ":INFO:Advancing CG variables in time" << std::endl;

		if(nHistory > 1)
			ierr = Integrator->integrate(Mesoscopic->Get_Coords(), Mesoscopic->Get_Velocities()); // call Pade->integrate method
		else {
				
			ierr = Integrator->integrate(Mesoscopic->Get_Coords(),  Mesoscopic->Get_Velocities());
                        CHKERRQ(ierr);
		}	

		// Fine-grain (recover atomistic configuration)
		for(int dim = 0; dim < Microscopic->Get_Dim(); dim++) {

				ierr = VecCopy(Microscopic->Get_Coords()[dim], Microscopic->Get_pCoords()[dim]);
				CHKERRQ(ierr);

		}

		ierr = (Microscopic->mapping)(Mesoscopic->Get_Coords(), Mesoscopic->Get_pCoords(), *this, DmsBase::fpLog);
		CHKERRQ(ierr);

		auto topIndices = Microscopic->getTopIndices();

        	if (topIndices.size() > 0) {
                	fpLog << getTime() << "Initializing constrained recovery" << std::endl;

			//for(auto nchains = 0; nchains < Microscopic->getChains(); nchains++) {
 
	                ierr = consOptim(Microscopic->Get_Coords(), topIndices, Microscopic->getEqLength(), scale);

                	CHKERRQ(ierr);
        	}

		// save CG state
		for(int dim = 0; dim < Microscopic->Get_Dim(); dim++) {
			
			ierr = VecCopy(Mesoscopic->Get_Coords()[dim], Mesoscopic->Get_pCoords()[dim]);
			CHKERRQ(ierr);
		}

                // Pade: save CG state at t = 0
		
                for(int dim = 0; dim < Mesoscopic->Get_Dim(); dim++) {

                        ierr = VecCopy(Mesoscopic->Get_Coords()[dim], Mesoscopic->getCGTensor()[2][dim]);

                        for(int fr = Mesoscopic->getCGTensor().size() - 1; fr > 2; fr--) {
                                ierr = VecCopy(Mesoscopic->getCGTensor()[fr-1][dim], Mesoscopic->getCGTensor()[fr][dim]);
                                CHKERRQ(ierr);
                        }
                }
		

		fpLog << getTime() << ":INFO:Syncing GROMACS with DMS" << std::endl;
		// Done with DMS computations! Now update the GROMACS t_state to re-initiate the MD phase
		Microscopic->Sync_MD_fromDMS(this);

		fpLog << getTime() << ":INFO:Completed CG time step " << timeStep << ". Resuming micro(Gromacs) phase. " << std::endl;

		std::vector< char* > keys(Microscopic->Get_Dim());
		keys[0] = "/CGx";
		keys[1] = "/CGy";
		keys[2] = "/CGz";

		writePetsc(Mesoscopic->Get_Coords(), keys, timeStep, &viewer);
	}

	PetscFunctionReturn(ierr);
}

PetscErrorCode DmsBase::clean() {
	PetscFunctionBegin;

	delete Microscopic;
	delete Mesoscopic;
	delete Integrator;

	for(auto it = mesoMicroVec.begin(); it != mesoMicroVec.end(); it++)
		ierr = MatDestroy(&(*it));
	CHKERRQ(ierr);

	for(auto it = microMesoVec.begin(); it != microMesoVec.end(); it++)
		ierr = MatDestroy(&(*it));
	CHKERRQ(ierr);

	for(auto it = kernelVec.begin(); it != kernelVec.end(); it++)
        	ierr = MatDestroy(&(*it));
    	CHKERRQ(ierr);

	ierr = MatDestroy(&microMesoMap);
        CHKERRQ(ierr);

	ierr = MatDestroy(&mesoMicroMap);
        CHKERRQ(ierr);

	ierr = MatDestroy(&kernel);
        CHKERRQ(ierr);
    
	PetscFunctionReturn(ierr);
}

DmsBase::~DmsBase() {
	PetscFunctionBegin;

	// So, check to see if micro or meso objects should be deleted on all procs
	if(comm == PETSC_COMM_WORLD)
		ierr = clean();
	else
		if(!mpiRank)
			ierr = clean();

	if(!mpiRank) {
		fpLog.close();
		closePetsc(&viewer);
	}
}


int constructDmsCoords(dmsBasePtr swm) {
        PetscFunctionBegin;
        PetscErrorCode ierr;

        if(!reinterpret_cast<DmsBase*>(swm)->getRank()) {

	    DmsBase* dmsBase = reinterpret_cast<DmsBase*>(swm);

            ierr = dmsBase->Microscopic->Sync_DMS_fromMD(dmsBase);
            CHKERRQ(ierr);

            ierr = dmsBase->constructCoords();
            CHKERRQ(ierr);

	    ierr = dmsBase->constructVelocities();
            CHKERRQ(ierr);
        }

        PetscFunctionReturn(ierr);
}

/*
int constructDmsVelocs(dmsBasePtr swm) {
        PetscFunctionBegin;
        PetscErrorCode ierr;

        if(!reinterpret_cast<DmsBase*>(swm)->getRank()) {

                DmsBase* dmsBase = reinterpret_cast<DmsBase*>(swm);
                ierr = dmsBase->Microscopic->Sync_DMS_fromMD(dmsBase);
                CHKERRQ(ierr);

                ierr = dmsBase->updateRef();
                CHKERRQ(ierr);

                ierr = dmsBase->Mesoscopic->Construct_Coords(
                       dmsBase->Microscopic->Get_Velocs(),
                       dmsBase->Microscopic->Get_pVelocs(),
                       dmsBase);

                CHKERRQ(ierr);
        }

        PetscFunctionReturn(ierr);
}
*/

dmsBasePtr newDmsBase(const t_state* state, const t_mdatoms* mdatoms,
                      const gmx_mtop_t* top, const t_inputrec* ir, gmx_int64_t dim,
                      gmx_int64_t cgDim, int nCG, gmx_int64_t freq, const real dt, 
		      const gmx_int64_t t0, MPI_Comm comm, const int mSteps, const float scale, 
		      const PetscInt nHist, const PetscInt ssIndex, const PetscInt nss, char const* cgMeth, 
		      char* userRef, char* topFname, char* selFname) {

    std::string cgMethod = cgMeth;
    return reinterpret_cast<void*>(new DmsBase(state, mdatoms, top, ir, dim, cgDim, nCG, freq, dt, t0, comm, mSteps, 
				   scale, nHist, ssIndex, nss, cgMethod, userRef, topFname, selFname));
}

void delDmsBase(dmsBasePtr swm) {
    delete reinterpret_cast<DmsBase*>(swm);
}

int dmsCGStep(dmsBasePtr swm, gmx_int64_t step) {
    return reinterpret_cast<DmsBase*>(swm)->cgStep(step);
}

PetscErrorCode DmsBase::constructCoords() {
        /* This function constructs SWM coords, velocities, and forces, depending on the arguments
         * supplied. The three CG-construction function pointers in Mesoscopic should all point to
         * this function.
         */
        PetscFunctionBegin;

        fpLog << getTime() << ":INFO:Constructing CG coordinates" << std::endl;

        (Mesoscopic->mapping)(Microscopic->Get_Coords(), Microscopic->Get_pCoords(), *this, DmsBase::fpLog);

        PetscFunctionReturn(ierr);
}

std::vector<PetscScalar> DmsBase::compCentOfMass(const CVec& Coords) {
        PetscFunctionBegin;

        fpLog << getTime() << ":INFO:Computing center of mass" << std::endl;

        std::vector<PetscScalar> COM(Coords.size());
        PetscScalar TotalMass;

        VecSum(Microscopic->getMass(), &TotalMass);

        for(int dim = 0; dim < Coords.size(); dim++) {
                ierr = VecDot(Microscopic->getMass(), Coords[dim], &COM[dim]);
                DMS_CHKERRQ(ierr);

                COM[dim] /= TotalMass;

        }

 
        PetscFunctionReturn(COM);
}

std::vector<PetscScalar> DmsBase::getBox(const CVec& Coords) const {
        PetscFunctionBegin;

        PetscInt Dim = Microscopic->Get_Dim();
        std::vector<PetscScalar> Max_Box(Dim), Min_Box(Dim);

        for(int dim = 0; dim < Dim; dim++) {
                VecMax(Coords[dim], NULL, Max_Box.data() + dim);
                VecMin(Coords[dim], NULL, Min_Box.data() + dim);

                Max_Box[dim] = Max_Box[dim] * 1.1;
                Min_Box[dim] = Min_Box[dim] * 1.1;
        }

        std::vector<PetscScalar> Box(Dim);

        std::transform(Max_Box.begin(), Max_Box.end(), Min_Box.begin(), Box.begin(),
                        std::minus<PetscScalar>());

        PetscFunctionReturn(Box);
}

PetscErrorCode DmsBase::constructVelocities() {
        /* This function constructs SWM coords, velocities, and forces, depending on the arguments
         * supplied. The three CG-construction function pointers in Mesoscopic should all point to
         * this function.
         */
        PetscFunctionBegin;

	fpLog << getTime() << ":INFO:Computing CG velocities " << std::endl;

        for(int dim = 0; dim < Mesoscopic->Get_Dim(); dim++) {

                ierr = VecCopy(Mesoscopic->Get_Coords()[dim], Mesoscopic->Get_Velocities()[dim]);
                CHKERRQ(ierr);

                ierr = VecAXPY(Mesoscopic->Get_Velocities()[dim], -1.0, Mesoscopic->Get_pCoords()[dim]);
                CHKERRQ(ierr);

                ierr = VecScale(Mesoscopic->Get_Velocities()[dim], 1.0 / Microscopic->getLength() );
                CHKERRQ(ierr);

                CHKERRQ(ierr);
        }

        PetscFunctionReturn(ierr);
}

PetscErrorCode DmsBase::constructForces() {
        PetscFunctionBegin;

        fpLog << getTime() << ":INFO:Computing CG forces " << std::endl;

        for(int dim = 0; dim < Mesoscopic->Get_Dim(); dim++) {

                ierr = VecCopy(Mesoscopic->Get_Velocities()[dim], Mesoscopic->Get_Forces()[dim]);
                CHKERRQ(ierr);

                ierr = VecAXPY(Mesoscopic->Get_Forces()[dim], -1.0, Mesoscopic->Get_pVelocities()[dim]);
                CHKERRQ(ierr);

                ierr = VecScale(Mesoscopic->Get_Forces()[dim], 1.0 / Microscopic->getLength() );
                CHKERRQ(ierr);

                CHKERRQ(ierr);
        }

        PetscFunctionReturn(ierr);
}

std::string DmsBase::getTime() {
  PetscFunctionBegin;

  time_t rawtime;
  struct tm *timeinfo;
  char *buffer = new char[PETSC_MAX_PATH_LEN];

  time (&rawtime);
  timeinfo = localtime (&rawtime);

  strftime (buffer,PETSC_MAX_PATH_LEN,"%F %R:%S ",timeinfo);

  std::string time_str(buffer);
  PetscFunctionReturn(time_str);
}

PetscErrorCode DmsBase::constructRef() {

	PetscFunctionBegin;
	PetscErrorCode ierr;
/*
	if(DmsBase::userSuppliedRef()) {
		try {
			fp = fstream(DmsBase::userSuppliedRef(), std::ios::in);
			PetscInt count = 0;
			std::string line;

			fp = gotoLine(fp, 2);

			for(auto i = 0; i < DmsBase::getNatoms(); i++) {
				getline(fp, line);
				std::vector<std::string> elems = split(line, ' ');

				PetscInt dim = 0;

	        	for(auto it = elems.begin(); it != elems.end(); it++)
	                if (it->size() >= 5) { // remove white space /resid / resnum ... only coords should remain here
	                        
	                       	PescScalar val = static_cast<PetscScalar>(*it);
							ierr = VecSetValues(Microscopic->Get_RefCoords()[dim++], 1, &i, &val, INSERT_VALUES);
							CHKERRQ(ierr);

							if(dim == 2)
								break;
					}
			}
		}
		catch (int e) {
			std::cout << "An exception has occured in reading ref config file. Exception Nr. " << e << std::endl;
		}

	}
	else {
		for(int dim = 0; dim < Coords.size(); dim++) {
			ierr = VecCopy(Coords[dim], Microscopic->Get_RefCoords()[dim]);
			CHKERRQ(ierr);
		}
	}
*/

	for(int dim = 0; dim < (Microscopic->Get_RefCoords()).size(); dim++) {
                        ierr = VecCopy(Microscopic->Get_Coords()[dim], Microscopic->Get_RefCoords()[dim]);
                        CHKERRQ(ierr);
	}

	for(int dim = 0; dim < Microscopic->Get_Dim(); dim++) {

			ierr = VecCopy(Microscopic->Get_Coords()[dim], Microscopic->Get_RefCoords()[dim]);
			CHKERRQ(ierr);

			ierr = VecAssemblyBegin(Microscopic->Get_RefCoords()[dim]);
			CHKERRQ(ierr);
			ierr = VecAssemblyEnd(Microscopic->Get_RefCoords()[dim]);
			CHKERRQ(ierr);
	}

	PetscFunctionReturn(ierr);
}
