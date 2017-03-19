static char help[] = "Solve the nonlinear schroedinger equation\n\n";


#define _USE_MATH_DEFINES
#include <petscksp.h>
#include <iostream>
#include <cmath>
#include <math.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  double t1, t2;
  double ta = 0.0;
  double te = 1.0;
  double dt = 0.01;

  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt N = 16;
  PetscInt       i,Ii,Istart,Iend;
  PetscScalar    val;
  PetscScalar h, r, lambda;

  Vec            x, u, b;
  Mat            A;
  KSP            ksp;

  PetscInitialize(&argc,&argv,(char*)0,help);
  t1 = MPI_Wtime();
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
  ierr = VecSetSizes(u,PETSC_DECIDE,N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&b);CHKERRQ(ierr);

  h = 2*PETSC_PI / ( (PetscScalar) (N + 1) );
  r = dt / (h*h);
  lambda = 1.0;

  for (i=0; i<N; i++) {
    val = PetscExpComplex( PETSC_i * (((PetscScalar) (i + 1))*h - PETSC_PI));
    ierr = VecSetValues(u,1,&i,&val,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(u);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(u);CHKERRQ(ierr);
  //ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);


  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);


  ierr = MatMPIAIJSetPreallocation(A,3,NULL,3,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A,3,NULL);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation(A,1,3,NULL);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);

  val = r/2.0;
  int last = N-1;
  int first = 0;
  PetscScalar ab;

  ierr = MatSetValues(A,1,&last,1,&first,&val,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValues(A,1,&first,1,&last,&val,INSERT_VALUES);CHKERRQ(ierr);
  for (Ii=Istart; Ii<Iend; Ii++) {
    val = r/2.0;
    i=Ii+1;if (i<N) ierr = MatSetValues(A,1,&Ii,1,&i,&val,INSERT_VALUES);CHKERRQ(ierr);
    i=Ii-1;if (i>=0) ierr = MatSetValues(A,1,&Ii,1,&i,&val,INSERT_VALUES);CHKERRQ(ierr);
    val = 1.0;
    ierr = MatSetValues(A,1,&Ii,1,&Ii,&val,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);


  ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATIONS,PETSC_FALSE);CHKERRQ(ierr);
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);

  while (ta < te) {
      for (Ii=Istart; Ii<Iend; Ii++) {
        VecGetValues(u,1,&Ii,&ab);
        val = 1.0 - r - lambda*dt/2.0 * PetscAbsComplex(ab)*PetscAbsComplex(ab);
        ierr = MatSetValues(A,1,&Ii,1,&Ii,&val,INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

      ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
      ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
      ierr = KSPSolve(ksp,u,x);CHKERRQ(ierr);
      ta += dt;
  }


  t2 = MPI_Wtime();
  std::cout << t2 - t1 << "\n";




  //ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  //ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}
