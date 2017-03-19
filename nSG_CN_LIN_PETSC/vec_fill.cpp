static char help[] = "Builds a parallel vector with 1 component on the first processor, 2 on the second, etc.\n\
  Then each processor adds one to all elements except the last rank.\n\n";

#include <petscvec.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       i,N;
  PetscScalar    one = 1.0;
  Vec            x;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);


  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,16);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecGetSize(x,&N);CHKERRQ(ierr);

  
  for (i=0; i<N; i++) {
    one = i;
    ierr = VecSetValues(x,1,&i,&one,INSERT_VALUES);CHKERRQ(ierr);
  }


  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}
