
#include "globalVariables.cuh"

typedef double REAL;
typedef struct REAL3 { REAL x, y, t; } REAL3;

typedef struct stateVar {
	REAL *u, *v;
} stateVar;

typedef struct sliceVar {
	REAL *ux, *uy, *ut, *vx, *vy, *vt;
} sliceVar;

typedef struct vec5dyn 
	{ float x, y, vx, vy, t; } vec5dyn;

typedef struct advVar {
	REAL *pp, *mp, *mm, *pm;
} advVar;

typedef struct electrodeVar {
	REAL e0, e1;
} electrodeVar;

typedef struct velocity {
	REAL x, y;
} velocity;

typedef struct fileVar {
	char read[100], readcsv[100], p1D[100], p2D[100], p2DG[100], tip1[100], tip2[100], sym[100],
		contour1[100], contour2[100], param1[100], param2[100];
} fileVar;

typedef struct paramVar {

	int nx;
	int ny;
	int memSize;
	REAL Lx, Ly, hx, hy;
	REAL dt;
	REAL diff_par, diff_per;
	REAL Dxx, Dyy, Dxy;
	REAL rx, ry, rxy, rbx, rby;
	REAL rscale;
	REAL invdx, invdy;
	int sample, count;
	REAL physicalTime, physicalTimeLim;
	int stimPeriod;
	REAL stimMag;
	int eSize;
	int2 point;
	int nc;
	REAL rdomTrapz, rdomStim, rdomAPD;
	REAL stcx, stcy;
	float2 pointStim;
	int savePackage;
	float tiempo;
	REAL degrad;
	REAL boundaryVal;

	REAL Uth;
	REAL tau_e;
	REAL tau_n;
	REAL e_h;
	REAL e_n;
	REAL e_star;
	REAL Re;
	REAL M_d;
	REAL expRe;
	REAL tc;

	int itPerFrame;
	int tipOffsetX;
	int tipOffsetY;
	float minVarColor;
	float maxVarColor;

	int wnx;
	int wny;
	float uMax;
	float uMin;
	float vMax;
	float vMin;

	bool animate;
	bool recordTip;
	bool recordContour;
	bool stimulate;
	bool reduceSym;
	bool reduceSymStart;
	bool clock, counterclock;
	bool firstIterContour;
	bool firstFPS;

	bool circle;
	bool neumannBC;
	bool gateDiff;
	bool bilinear;
	bool anisotropy;
	bool tipGrad;

	bool load;
	bool save;

} paramVar;
