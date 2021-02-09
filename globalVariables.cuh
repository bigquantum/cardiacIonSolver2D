

#define STEP(a,b) (a >= b)
#define I2D(nn,i,j) (((nn)*(j)) + i)
#define VOLT2PIX(val,vMIN,vMAX,N) ( ((val)-(vMIN))*(N)/((vMAX)-(vMIN)) ) // Transform voltage values to pixel values

/////////////
// Constants
/////////////

// TIPVECSIZE is the maximum number of structs to insert
#define TIPVECSIZE 500000

// Macro for 2D Finite Difference
#define BLOCK_DIM_X (16)
#define BLOCK_DIM_Y (16)
#define BLOCKSIZE_1D (64)
#define GRIDSIZE_1D (256)

// Linear solver parameters
#define NSYM (3)
#define NR_END (1)
#define FREE_ARG char*
#define TINY 1.0e-20

#define MACHINE_EPS (2.220446049250313e-16)
#define pi (3.14159265359)

/*========================================================================
 * Model parameters defined as macros
 *========================================================================
*/

// #define Uth (2.0)

// #define tau_e (6.5)
// #define tau_n (270.0)
// #define e_h (3.0)
// #define e_n (1.0)
// #define e_star (1.5415)
// #define Re (0.4)
// #define M_d (7)
// #define expRe (expf(-Re))
// #define tc (1.0)

// #define tau_e (6.5)
// #define tau_n (270.0)
// #define e_h (3.0)
// #define e_n (1.0)
// #define e_star (1.5415)
// #define Re (0.4)
// #define M_d (4)
// #define expRe (expf(-Re))
// #define tc (1.0)