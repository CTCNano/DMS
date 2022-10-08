#include "classes.h"
#include "mtop_util.h"
#include "../gromacs/gmxpreprocess/grompp-impl.h"

PetscInt compNumSolMol(const gmx_mtop_t* mdTop) {
	// Finds the number of nonsolvent / non-ion atoms ... must extend the ion resnames
	// beyond NA/CL ...
	// It is worth noting that Gromacs does NOT store the indices of the non-solvent atoms contiguously
	// in memory. So we need to find these indices ... this seems like a good start for subsystem decomposition.

	PetscFunctionBegin;
	t_atom *atom;
        char* resname, *atomname;
        int resnum, i;
	PetscInt count = 0;

        gmx_mtop_atomloop_all_t aloop = gmx_mtop_atomloop_all_init(mdTop);

        while (gmx_mtop_atomloop_all_next(aloop, &i, &atom)) {

        	gmx_mtop_atomloop_all_names(aloop, &atomname, &resnum, &resname);

                if(strncmp(resname, "SOL", 3) && strncmp(resname, "NA", 2) && strncmp(resname, "CL", 2) && strncmp(resname, "GRA", 3))
                	count++;
	}

	PetscFunctionReturn(count);
}

bool subsystemDecomp(const gmx_mtop_t* mdTop) {
    int                     i, resnr;
    gmx_mtop_atomloop_all_t aloop;
    t_atom                 *atom;
    char                   *atomname, *resname;

    aloop = gmx_mtop_atomloop_all_init(mdTop);

    while (gmx_mtop_atomloop_all_next(aloop, &i, &atom)) {
        gmx_mtop_atomloop_all_names(aloop, &atomname, &resnr, &resname);
	std::cout << resname << " " << resnr << std::endl;
    }
}

std::fstream& GotoLine(std::fstream& file, unsigned int num){
    file.seekg(std::ios::beg);
    for(int i=0; i < num - 1; ++i){
        file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    }
    return file;
}

PetscErrorCode Micro_state::setupSubsystem(char* selFname) {

    PetscFunctionBegin;
    PetscErrorCode ierr;

    if (!selFname) {

	DOF_local = DOF;
	ssIndices = NULL;

    }
    else {

	std::cout << "Reading subsystem indices from " << selFname << " for subsystem " << ssIndex << std::endl;

	PetscInt count = 0;
    	std::fstream fp;
    	fp.open(selFname, std::ios::in);

	GotoLine(fp, ssIndex + 1);

	PetscInt index1, index2;

	fp >> index1 >> index2;

	DOF_local = index2 - index1 + 1;
    	ssIndices = new PetscInt[DOF_local];

	for(auto j = index1; j <= index2; j++)	    
		ssIndices[count++] = j - 1; // the *-1* is because the user supplies the indices starting from 1 and not 0.

    	fp.close();
    }

    PetscFunctionReturn(ierr);

}


PetscErrorCode Micro_state::setupRefTop(char* topFname, DmsBase* Dbase) {

    PetscFunctionBegin;
    PetscErrorCode ierr;

    ierr = Sync_DMS_fromMD(Dbase);
    DMS_CHKERRQ(ierr);

    std::cout << "Done syncing DMS from MD" << std::endl;

    for(int dim = 0; dim < Dim; dim++) {

		ierr = VecCopy(Get_Coords()[dim], Get_RefCoords()[dim]);
		DMS_CHKERRQ(ierr);

		ierr = VecAssemblyBegin(Get_RefCoords()[dim]);
		DMS_CHKERRQ(ierr);
		ierr = VecAssemblyEnd(Get_RefCoords()[dim]);
		DMS_CHKERRQ(ierr);
     }

	if(topFname) {

	    topIndices = dmsReadTop(topFname);

	    ierr = VecCreateSeq(PETSC_COMM_SELF, topIndices.size(), &eqLength);
	    DMS_CHKERRQ(ierr);

	    PetscInt    *index = new PetscInt[topIndices.size()];
            PetscScalar *atomCoordsOne = new PetscScalar[topIndices.size()],
     		        *atomCoordsTwo = new PetscScalar[topIndices.size()];

	    ierr = VecZeroEntries(eqLength);
	    DMS_CHKERRQ(ierr);

	    for(auto dim = 0; dim < Dim; dim++) {

		    Vec tmp;

		    for(int i = 0; i < topIndices.size(); i++)
        			index[i] = topIndices[i][0];

		    ierr = VecGetValues(Coords[dim], topIndices.size(), index, atomCoordsOne);
		    CHKERRQ(ierr);

		    for(auto i = 0; i < topIndices.size(); i++)
			index[i] = topIndices[i][1];

		    ierr = VecGetValues(Coords[dim], topIndices.size(), index, atomCoordsTwo);
		    CHKERRQ(ierr);

		    for(auto i = 0; i < topIndices.size(); i++)
        			atomCoordsTwo[i] = pow(atomCoordsOne[i] - atomCoordsTwo[i], 2.0);

        	    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, topIndices.size(), atomCoordsTwo, &tmp);
		    CHKERRQ(ierr);

        	    ierr = VecAXPY(eqLength, 1.0, tmp);
		    CHKERRQ(ierr);

        	    ierr = VecDestroy(&tmp);
		    CHKERRQ(ierr);
 		}

	    delete[] index;
	    delete[] atomCoordsOne;
	    delete[] atomCoordsTwo;
    }

    PetscFunctionReturn(ierr);
}

Micro_state::Micro_state(const t_state* state, const t_mdatoms* mdatoms,
		const gmx_mtop_t* top, const t_inputrec* ir, PetscInt Dim, MPI_Comm Comm, 
		ptrMap ptr_func, int microSteps, const real dt, 
		PetscInt nSS, PetscInt ssI, DmsBase* Dbase, char* topFname, char* sFname) : Dim(Dim) {

	// TODO: atomic forces must be included
	// TODO: destroy vec/mat PETSC objects before allocating new ones for copy const and operator=
	PetscFunctionBeginUser;

	mode = NULL;

	COMM = Comm;
	Coords.resize(Dim);
	pCoords.resize(Dim);
	Ref_Coords.resize(Dim);
	Velocities.resize(Dim);
	pVelocities.resize(Dim);
	Forces.resize(Dim);

	MD_state = state;
	MD_top = top;
	tmdatoms = mdatoms;
	mapping = ptr_func;
	DOF = compNumSolMol(top);
	numSS = nSS;
	ssIndex = ssI;
	selFname = sFname;

	atomIndices.resize(DOF);

	mdLength = ir->delta_t * microSteps;
	std::cout << "Micro length is " << mdLength << " ps" << std::endl;

	MPI_Comm_size(PETSC_COMM_WORLD, &MPI_Size);
	MPI_Comm_rank(PETSC_COMM_WORLD, &MPI_Rank);

	std::cout << "Preallocating memory for PETSc vector ..." << std::endl;

	ierr = setupSubsystem(selFname);	
	DMS_CHKERRQ(ierr);

	std::cout << "Found a total of " << DOF << " atoms, with a local of " << DOF_local << " atoms per subsystem " << ssIndex << std::endl;

	if(COMM == PETSC_COMM_WORLD)
		for(int dim = 0; dim < Dim; dim++) {
			ierr = VecCreateMPI(COMM, PETSC_DECIDE, DOF_local, Coords.data() + dim);
			ierr = VecCreateMPI(COMM, PETSC_DECIDE, DOF_local, pCoords.data() + dim);
			ierr = VecCreateMPI(COMM, PETSC_DECIDE, DOF_local, Velocities.data() + dim);
			ierr = VecCreateMPI(COMM, PETSC_DECIDE, DOF_local, pVelocities.data() + dim);
			ierr = VecCreateMPI(COMM, PETSC_DECIDE, DOF_local, Forces.data() + dim);
			ierr = VecCreateMPI(COMM, PETSC_DECIDE, DOF_local, Ref_Coords.data() + dim);
		}
	else
    		for(int dim = 0; dim < Dim; dim++) {
			ierr = VecCreateSeq(COMM, DOF_local, Coords.data() + dim);
			ierr = VecCreateSeq(COMM, DOF_local, pCoords.data() + dim);
			ierr = VecCreateSeq(COMM, DOF_local, Velocities.data() + dim);
			ierr = VecCreateSeq(COMM, DOF_local, pVelocities.data() + dim);
			ierr = VecCreateSeq(COMM, DOF_local, Forces.data() + dim);
			ierr = VecCreateSeq(COMM, DOF_local, Ref_Coords.data() + dim);
		}


	if(COMM == MPI_COMM_WORLD)
		ierr = VecCreateMPI(COMM, PETSC_DECIDE, DOF_local, &Mass);
	else
		ierr = VecCreateSeq(COMM, DOF_local, &Mass);

	DMS_CHKERRQ(ierr);

	std::cout << "Reading atomic masses for selected subsystem " << ssIndex << std::endl;

	// TODO: not quite efficient (block assembly is better ...) but it's done only once so it's OK fo now
	if(!MPI_Rank) {
		// I took this piece of code from grompp.c
		PetscScalar mass;
		t_atom         *atom;
		char* resname, *atomname;
		int resnum, atomindex, count = 0, ai;

		//gmx_mtop_atomlookup_t alook;
		//alook = gmx_mtop_atomlookup_init(MD_top);
		gmx_mtop_atomloop_all_t aloop = gmx_mtop_atomloop_all_init(MD_top);

		while (gmx_mtop_atomloop_all_next(aloop, &atomindex, &atom)) {

			gmx_mtop_atomloop_all_names(aloop, &atomname, &resnum, &resname);
			//gmx_mtop_atomnr_to_atom(alook, ai, &atom);

			if(selFname) {
 
				if(atomindex >= ssIndices[0] && atomindex <= ssIndices[DOF_local-1]) {

					mass = static_cast<PetscScalar>(atom->m);
                                	ierr = VecSetValues(Mass, 1, &count, &mass, INSERT_VALUES);
                                	atomIndices[count] = count;
                                	count++;
				}
		
			}
			else
				if(strncmp(resname, "SOL", 3) && strncmp(resname, "NA", 2) && strncmp(resname, "CL", 2) && strncmp(resname, "GRA", 3)) { 
					mass = static_cast<PetscScalar>(atom->m);
		    			ierr = VecSetValues(Mass, 1, &count, &mass, INSERT_VALUES);
					atomIndices[count] = count;
					count++;
				}
		}

		//gmx_mtop_atomloop_all_destroy(aloop);
	}

	ierr = VecAssemblyBegin(Mass);
	DMS_CHKERRQ(ierr);

	ierr = VecAssemblyEnd(Mass);
	DMS_CHKERRQ(ierr);

	if(!MPI_Rank) {
		ierr = setupRefTop(topFname, Dbase);
		DMS_CHKERRQ(ierr);
	}

    DMS_CHKERRQ(ierr);
}

Micro_state::Micro_state(const Micro_state& micro_class) : COMM(micro_class.COMM),
		DOF(micro_class.DOF), DOF_local(micro_class.DOF_local) {

	PetscFunctionBegin;

	Coords.resize(Dim);
	pCoords.resize(Dim);
	Velocities.resize(Dim);
	Forces.resize(Dim);

	mapping = micro_class.mapping;
	MD_top = micro_class.Get_MD_top();
	MD_state = micro_class.Get_MD_state();

	for(int dim = 0; dim < Dim; dim++)
		if(COMM == PETSC_COMM_WORLD) {
			ierr = VecCreateMPI(COMM, PETSC_DECIDE, DOF_local, Coords.data() + dim);
			DMS_CHKERRQ(ierr);
			ierr = VecCopy(micro_class.Get_Coords()[dim], Get_Coords()[dim]);
			DMS_CHKERRQ(ierr);
			ierr = VecAssemblyBegin(Get_Coords()[dim]);
			DMS_CHKERRQ(ierr);
			ierr = VecAssemblyEnd(Get_Coords()[dim]);
			DMS_CHKERRQ(ierr);

			ierr = VecCreateMPI(COMM, PETSC_DECIDE, DOF_local, pCoords.data() + dim);
			DMS_CHKERRQ(ierr);
			ierr = VecCopy(micro_class.Get_pCoords()[dim], Get_pCoords()[dim]);
			DMS_CHKERRQ(ierr);
			ierr = VecAssemblyBegin(Get_pCoords()[dim]);
			DMS_CHKERRQ(ierr);
			ierr = VecAssemblyEnd(Get_pCoords()[dim]);
			DMS_CHKERRQ(ierr);

			ierr = VecCreateMPI(COMM, PETSC_DECIDE, DOF_local, Velocities.data() + dim);
			DMS_CHKERRQ(ierr);
			ierr = VecCopy(micro_class.Get_Velocities()[dim], Get_Velocities()[dim]);
			DMS_CHKERRQ(ierr);
			ierr = VecAssemblyBegin(Get_Velocities()[dim]);
			DMS_CHKERRQ(ierr);
			ierr = VecAssemblyEnd(Get_Velocities()[dim]);
			DMS_CHKERRQ(ierr);

			ierr = VecCreateMPI(COMM, PETSC_DECIDE, DOF_local, Forces.data() + dim);
			DMS_CHKERRQ(ierr);
			ierr = VecCopy(micro_class.Get_Forces()[dim], Get_Forces()[dim]);
			DMS_CHKERRQ(ierr);
			ierr = VecAssemblyBegin(Get_Forces()[dim]);
			DMS_CHKERRQ(ierr);
			ierr = VecAssemblyEnd(Get_Forces()[dim]);
			DMS_CHKERRQ(ierr);
		}
		else {
			ierr = VecCreateSeq(COMM, DOF_local, Coords.data() + dim);
			DMS_CHKERRQ(ierr);
			ierr = VecCopy(micro_class.Get_Coords()[dim], Get_Coords()[dim]);
			DMS_CHKERRQ(ierr);

			ierr = VecCreateSeq(COMM, DOF_local, pCoords.data() + dim);
			DMS_CHKERRQ(ierr);
			ierr = VecCopy(micro_class.Get_pCoords()[dim], Get_pCoords()[dim]);
			DMS_CHKERRQ(ierr);

			ierr = VecCreateSeq(COMM, DOF_local, Velocities.data() + dim);
			DMS_CHKERRQ(ierr);
			ierr = VecCopy(micro_class.Get_Velocities()[dim], Get_Velocities()[dim]);
			DMS_CHKERRQ(ierr);

			ierr = VecCreateSeq(COMM, DOF_local, Forces.data() + dim);
			DMS_CHKERRQ(ierr);
			ierr = VecCopy(micro_class.Get_Forces()[dim], Get_Forces()[dim]);
			DMS_CHKERRQ(ierr);
		}
}

Micro_state &Micro_state::operator=(const Micro_state& micro_class) {
	// TODO: Fix this function
	PetscFunctionBegin;

	COMM = micro_class.COMM;
	DOF = micro_class.DOF;
	Dim = micro_class.Dim;

	mapping = micro_class.mapping;

	Coords.resize(Dim);
	Velocities.resize(Dim);
	Forces.resize(Dim);

	for(int dim = 0; dim < micro_class.Dim; dim++) {
		ierr = VecCreateMPI(COMM, PETSC_DECIDE, DOF_local, Coords.data() + dim);
		ierr = VecCopy(micro_class.Get_Coords()[dim], Get_Coords()[dim]);
		ierr = VecAssemblyBegin(Get_Coords()[dim]);
		ierr = VecAssemblyEnd(Get_Coords()[dim]);

		VecCreateMPI(COMM, PETSC_DECIDE, DOF_local, Velocities.data() + dim);
		VecCopy(micro_class.Get_Velocities()[dim], Get_Velocities()[dim]);
		VecAssemblyBegin(Get_Velocities()[dim]);
		VecAssemblyEnd(Get_Velocities()[dim]);

		VecCreateMPI(COMM, PETSC_DECIDE, DOF_local, Forces.data() + dim);
		VecCopy(micro_class.Get_Forces()[dim], Get_Forces()[dim]);
		VecAssemblyBegin(Get_Forces()[dim]);
		VecAssemblyEnd(Get_Forces()[dim]);
		}

	return *this;
}

PetscErrorCode Micro_state::Sync_DMS_fromMD(DmsBase* Dbase) {
	PetscFunctionBegin;

	std::cout << "Syncing DMS with GROMACS" << std::endl;

	// TODO: This can be made more efficient i.e. Values allocated once and Indices is constant
	std::vector< std::vector<PetscScalar> > Values(Dim), ValuesV(Dim);
	std::vector<PetscInt> Indices(DOF_local);

	for(int dim = 0; dim < Dim; dim++) {
		Values[dim].resize(DOF_local);
		ValuesV[dim].resize(DOF_local);
	}

	for(int dim = 0; dim < Dim; dim++) {

		if(!MPI_Rank) {

			t_atom         *atom;
                        char* resname, *atomname;
                        int resnum, atomindex, count = 0;

                        gmx_mtop_atomloop_all_t aloop = gmx_mtop_atomloop_all_init(MD_top);

			while(count < DOF_local) {

				if(selFname) {
                                          Values[dim][count] = MD_state->x[ssIndices[count]][dim];
					  Indices[count] = count++;
				}
                                else

                        		while (gmx_mtop_atomloop_all_next(aloop, &atomindex, &atom)) {

                               			gmx_mtop_atomloop_all_names(aloop, &atomname, &resnum, &resname);

                        			if(strncmp(resname, "SOL", 3) && strncmp(resname, "NA", 2) && strncmp(resname, "CL", 2) && strncmp(resname, "GRA", 3)) {

                                	  		Values[dim][count] = MD_state->x[atomindex][dim];
							ValuesV[dim][count] = MD_state->v[atomindex][dim];
							Indices[count] = count++;

						}
					}
			}

			ierr = VecSetValues(Get_Coords()[dim], DOF_local, Indices.data(), Values[dim].data(),
								 INSERT_VALUES);
			CHKERRQ(ierr);

			ierr = VecSetValues(Get_Velocities()[dim], DOF_local, Indices.data(), ValuesV[dim].data(),
							  	 INSERT_VALUES);
			CHKERRQ(ierr);
		}

		ierr = VecAssemblyBegin(Get_Coords()[dim]);
		CHKERRQ(ierr);
		ierr = VecAssemblyEnd(Get_Coords()[dim]);
		CHKERRQ(ierr);

		ierr = VecAssemblyBegin(Get_Velocities()[dim]);
		CHKERRQ(ierr);
		ierr = VecAssemblyEnd(Get_Velocities()[dim]);
		CHKERRQ(ierr);
	}

	PetscFunctionReturn(ierr);
}

PetscErrorCode Micro_state::Sync_MD_fromDMS(DmsBase* Dbase) {
	PetscFunctionBegin;

	std::cout << "Syncing GROMACS with DMS" << std::endl;
	PetscScalar *Coords_ptr, *Vels_ptr, *Forces_ptr;
	// TODO: Figure out how this is gonna work in parallel

	if(!MPI_Rank)
		for(int dim = 0; dim < Dim; dim++) {

				ierr = VecGetArray(Coords[dim], &Coords_ptr);
				CHKERRQ(ierr);

				ierr = VecGetArray(Velocities[dim], &Vels_ptr);
                		CHKERRQ(ierr);

				ierr = VecGetArray(Forces[dim], &Forces_ptr);
                                CHKERRQ(ierr);

				t_atom         *atom;
		                char* resname, *atomname;
                		int resnum, atomindex, count = 0;

                		gmx_mtop_atomloop_all_t aloop = gmx_mtop_atomloop_all_init(MD_top);
				//PetscScalar velocity = .0;

				while(count < DOF_local) {

					if (selFname)
                                        		MD_state->x[ssIndices[count]][dim] = Coords_ptr[count++];
					else 
                				while (gmx_mtop_atomloop_all_next(aloop, &atomindex, &atom)) {

                        				gmx_mtop_atomloop_all_names(aloop, &atomname, &resnum, &resname);

                        				if(strncmp(resname, "SOL", 3) && strncmp(resname, "NA", 2) && strncmp(resname, "CL", 2) && strncmp(resname, "GRA", 3)) {
								//velocity = Coords_ptr[count] - MD_state->x[atomindex][dim];

								if(mode)
									MD_state->x[atomindex][dim] += Coords_ptr[count++]; 
								else
                                					MD_state->x[atomindex][dim] = Coords_ptr[count++];
							}
							//else
								//if (!strncmp(resname, "SOL", 3) || !strncmp(resname, "NA", 2) || !strncmp(resname, "CL", 2))
								//	MD_state->x[atomindex][dim] += velocity;

                				}
				}

				// std::cout << "translated the system by " << velocity << std::endl;

				ierr = VecRestoreArray(Coords[dim], &Coords_ptr);
				CHKERRQ(ierr);

				ierr = VecRestoreArray(Velocities[dim], &Vels_ptr);
                                CHKERRQ(ierr);
			}

	PetscFunctionReturn(ierr);
}

Micro_state::~Micro_state() {
	PetscFunctionBegin;

	// no need for this unless you dont wanna allocate memory every CG_step

	for(int dim = 0; dim < Dim; dim++) {
			ierr = VecDestroy(Coords.data() + dim);
			DMS_CHKERRQ(ierr);

			ierr = VecDestroy(pCoords.data() + dim);
			DMS_CHKERRQ(ierr);

			ierr = VecDestroy(Velocities.data() + dim);
			DMS_CHKERRQ(ierr);

			ierr = VecDestroy(pVelocities.data() + dim);
                        DMS_CHKERRQ(ierr);

			ierr = VecDestroy(Forces.data() + dim);
			DMS_CHKERRQ(ierr);
		}

	delete[] ssIndices;
}
