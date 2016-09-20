#ifndef __LOADER_H_
#define __LOADER_H_

#include "geometry.h"

extern unsigned verticesNo;
extern Vertex* vertices;
extern unsigned int trianglesNo;
extern Triangle* triangles;

void panic(const char *fmt, ...);
void load_object(const char *filename);
float processgeo();

#endif