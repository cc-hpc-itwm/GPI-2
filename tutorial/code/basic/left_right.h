#ifndef LEFT_RIGHT_H
#define LEFT_RIGHT_H

#define RIGHT(iProc,nProc) ((iProc + nProc + 1) % nProc)
#define LEFT(iProc,nProc) ((iProc + nProc - 1) % nProc)

#endif
