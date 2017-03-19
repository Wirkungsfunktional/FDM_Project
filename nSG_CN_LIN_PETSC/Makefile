ALL: ex2
CFLAGS = 
FFLAGS = 
CPPFLAGS = 
FPPFLAGS = 
CLEANFILES = ex2

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex2: ex2.o chkopts
	${CLINKER} -o ex2 ex2.o ${PETSC_LIB}
	${RM} ex2.o

vec_fill: vec_fill.o chkopts
	${CLINKER} -o vec_fill vec_fill.o ${PETSC_LIB}
	${RM} vec_fill.o

compile: main.o chkopts
	${CLINKER} -o main main.o ${PETSC_LIB}
	${RM} main.o
