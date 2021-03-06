static char help[] = "Solve the nonlinear schroedinger equation\n\n";


#define _USE_MATH_DEFINES
#include <petscksp.h>
#include <iostream>
#include <cmath>
#include <math.h>
#include <stdlib.h>
#include "petscpf.h"


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  if (argc < 5) {
      std::cout << "Missing argument: N, dt, te, opt" << "\n";
      exit(1);
  }

  PetscInt N = atoi(argv[1]);
  double dt = strtod(argv[2], NULL);
  double te = strtod(argv[3], NULL);
  int opt = atoi(argv[4]);


  double t1, t2;
  double ta = 0.0;
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscInt loc_n;
  PetscInt       i,Ii,Istart,Iend;
  PetscScalar    val;
  PetscScalar h, r, lambda;
  int k;

  PetscInt ind[N];

  Vec            u, b, r_vec, ana_u;
  Mat            A;
  KSP            ksp;

  PetscInitialize(&argc,&argv,(char*)0,help);
  t1 = MPI_Wtime();
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
  ierr = VecSetSizes(u,PETSC_DECIDE,N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&r_vec);CHKERRQ(ierr);

  h = 2*PETSC_PI / ( (PetscScalar) (N) );
  r = PETSC_i * dt / (h*h);
  lambda = 1.0;

  ierr = VecGetLocalSize(u, &loc_n);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(u,&Istart,&Iend);CHKERRQ(ierr);
  PetscScalar data[loc_n];
  for (i=Istart, k = 0; i<Iend; i++, k++) {
    data[k] = PetscExpComplex( PETSC_i * (((PetscScalar) (i+1))*h - PETSC_PI));
    ind[k] = i;
  }

  ierr = VecSetValues(u,loc_n,ind,data,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(u);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(u);CHKERRQ(ierr);
  //ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /*--------------------------------------------------------------------------*/
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

  MatSetValues(A,1,&last,1,&first,&val,INSERT_VALUES);CHKERRQ(ierr);
  MatSetValues(A,1,&first,1,&last,&val,INSERT_VALUES);CHKERRQ(ierr);
  for (Ii=Istart; Ii<Iend; Ii++) {
    val = r/2.0;
    i=Ii+1;if (i<N) ierr = MatSetValues(A,1,&Ii,1,&i,&val,INSERT_VALUES);CHKERRQ(ierr);
    i=Ii-1;if (i>=0) ierr = MatSetValues(A,1,&Ii,1,&i,&val,INSERT_VALUES);CHKERRQ(ierr);
    val = -r;
    ierr = MatSetValues(A,1,&Ii,1,&Ii,&val,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //ierr = MatDuplicate(A,MAT_SHARE_NONZERO_PATTERN,&B);CHKERRQ(ierr);
  //ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);


  ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);


  while (ta < te) {
      ierr = VecSet(r_vec, -r);
      ierr = VecCopy(u,b);CHKERRQ(ierr);
      ierr = VecAbs(b);CHKERRQ(ierr);
      ierr = VecPointwiseMult(b, b,b);CHKERRQ(ierr);
      ierr = VecScale(b, PETSC_i * lambda*dt/2.0);CHKERRQ(ierr);
      ierr = VecAXPY(r_vec, -1.0, b);CHKERRQ(ierr);
      ierr = MatDiagonalSet(A,r_vec,INSERT_VALUES);CHKERRQ(ierr);
      //ierr = MatAYPX(B,1.0,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);

      ierr = MatMult(A,u,b);CHKERRQ(ierr);
      ierr = VecScale(b, -2.0);CHKERRQ(ierr);
      ierr = MatShift(A, -1.0);


      ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
      ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
      KSPSetTolerances(ksp,1.e-12,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);


      ierr = KSPSolve(ksp,b,b);CHKERRQ(ierr);
      ierr = VecAXPY(u,1.0,b);CHKERRQ(ierr);

      ta += dt;
  }


  t2 = MPI_Wtime();
  //std::cout << t2 - t1 << "\n";
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank );
  if (opt==1) {
          double err;
          ierr = VecDuplicate(u,&ana_u);CHKERRQ(ierr);
          ierr = VecGetOwnershipRange(ana_u,&Istart,&Iend);CHKERRQ(ierr);
          for (i=Istart, k = 0; i<Iend; i++, k++) { //np.exp(1.j*x + 1.j*(-l - 1)*t)
            data[k] = PetscExpComplex( PETSC_i*(((PetscScalar) (i+1))*h - PETSC_PI) - 2.0*PETSC_i*ta );
            ind[k] = i;
          }
          ierr = VecSetValues(ana_u,loc_n,ind,data,INSERT_VALUES);CHKERRQ(ierr);
          ierr = VecAssemblyBegin(ana_u);CHKERRQ(ierr);
          ierr = VecAssemblyEnd(ana_u);CHKERRQ(ierr);
          ierr = VecAXPY(u,-1.0,ana_u);CHKERRQ(ierr);
          ierr = VecNorm(u,NORM_INFINITY,&err);CHKERRQ(ierr);
    if (rank == 0 ) {
          std::cout << err << "\n";
      }
  } else {
      PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_IMPL);
      ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }


  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&r_vec);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();

  return 0;
}
