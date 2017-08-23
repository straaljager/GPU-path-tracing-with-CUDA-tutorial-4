#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <cfloat>
#include <string.h>
#include <assert.h>

#include "linear_math.h"
#include "Geometry.h"
#include "SceneLoader.h"


using std::string;

unsigned verticesNo = 0;
unsigned trianglesNo = 0;
Vertex* vertices = NULL;   // vertex list
Triangle* triangles = NULL;  // triangle list

 
struct face {                  
	std::vector<int> vertex;
	std::vector<int> texture;
	std::vector<int> normal;
};

std::vector<face> faces;

namespace enums {
	enum ColorComponent {
		Red = 0,
		Green = 1,
		Blue = 2
	};
}

using namespace enums;

// Rescale input objects to have this size...
const float MaxCoordAfterRescale = 1.2f;

// if some file cannot be found, panic and exit
void panic(const char *fmt, ...)
{
	static char message[131072];
	va_list ap;

	va_start(ap, fmt);
	vsnprintf(message, sizeof message, fmt, ap);
	printf(message); fflush(stdout);
	va_end(ap);

	exit(1);
}

struct TriangleMesh
{
	std::vector<Vec3f> verts;
	std::vector<Vec3i> faces;
	Vec3f bounding_box[2];   // mesh bounding box
};

void load_object(const char *filename)
{
	std::cout << "Loading object..." << std::endl;
	const char* edot = strrchr(filename, '.');
	if (edot) {
		edot++;

		// Stanford PLY models

		if (!strcmp(edot, "PLY") || !strcmp(edot, "ply")) { 
			// Only shadevis generated objects, not full blown parser!
			std::ifstream file(filename, std::ios::in);
			if (!file) {
				panic((string("Missing ") + string(filename)).c_str());
			}

			Vertex *pCurrentVertex = NULL;
			Triangle *pCurrentTriangle = NULL;

			string line;
			unsigned totalVertices, totalTriangles, lineNo = 0;
			bool inside = false;
			while (getline(file, line)) {
				lineNo++;
				if (!inside) {
					if (line.substr(0, 14) == "element vertex") {
						std::istringstream str(line);
						string word1;
						str >> word1;
						str >> word1;
						str >> totalVertices;
						vertices = (Vertex *)malloc(totalVertices*sizeof(Vertex));
						verticesNo = totalVertices;
						pCurrentVertex = vertices;
					}
					else if (line.substr(0, 12) == "element face") {
						std::istringstream str(line);
						string word1;
						str >> word1;
						str >> word1;
						str >> totalTriangles;
						triangles = (Triangle *)malloc(totalTriangles*sizeof(Triangle));
						trianglesNo = totalTriangles;
						pCurrentTriangle = triangles;
					}
					else if (line.substr(0, 10) == "end_header")
						inside = true;
				}
				else {
					if (totalVertices) {

						totalVertices--;
						float x, y, z;

						std::istringstream str_in(line);
						str_in >> x >> y >> z;

						pCurrentVertex->x = x;
						pCurrentVertex->y = y;
						pCurrentVertex->z = z;
						pCurrentVertex->_normal.x = 0.f; // not used for now, normals are computed on-the-fly by GPU 
						pCurrentVertex->_normal.y = 0.f; // not used
						pCurrentVertex->_normal.z = 0.f; // not used 

						pCurrentVertex++;
					}

					else if (totalTriangles) {

						totalTriangles--;
						unsigned dummy;
						float r, g, b;
						unsigned idx1, idx2, idx3; // vertex index
						std::istringstream str2(line);
						if (str2 >> dummy >> idx1 >> idx2 >> idx3)
						{
							// set rgb colour to white
							r = 255; g = 255; b = 255;

							pCurrentTriangle->_idx1 = idx1;
							pCurrentTriangle->_idx2 = idx2;
							pCurrentTriangle->_idx3 = idx3;
			
							Vertex *vertexA = &vertices[idx1];
							Vertex *vertexB = &vertices[idx2];
							Vertex *vertexC = &vertices[idx3];
							pCurrentTriangle++;
						}
					}
				}
			}
		}  // end of ply loader code

		////////////////////
		// basic OBJ loader
		////////////////////

		// the OBJ loader code based is an improved version of the obj code in  
		// http://www.keithlantz.net/2013/04/kd-tree-construction-using-the-surface-area-heuristic-stack-based-traversal-and-the-hyperplane-separation-theorem/
		// this code triangulates models with quads, n-gons and triangle fans

		else if (!strcmp(edot, "obj")) {

			std::cout << "loading OBJ model: " << filename;
			std::string filenamestring = filename;
			std::ifstream in(filenamestring.c_str());
			std::cout << filenamestring << "\n";

			if (!in.good())
			{
				std::cout << "ERROR: loading obj:(" << filename << ") file not found or not good" << "\n";
				system("PAUSE");
				exit(0);
			}

			Vertex *pCurrentVertex = NULL;
			Triangle *pCurrentTriangle = NULL;
			unsigned totalVertices, totalTriangles = 0;
			TriangleMesh mesh;

			/*
			char buffer[256], str[255];
			float f1, f2, f3;

			while (!in.getline(buffer, 255).eof()) // reads new line from file
			{
				buffer[255] = '\0'; // denotes end of stringbuffer
				sscanf_s(buffer, "%s", str, 255);

				//	std::cout << buffer << std::endl;


				// reading a vertex
				if (buffer[0] == 'v' && (buffer[1] == ' ' || buffer[1] == 32)){   /// vertices 
					if (sscanf(buffer, "v %f %f %f", &f1, &f2, &f3) == 3){
						mesh.verts.push_back(Vec3f(f1, f2, f3));
					}
					else{
						std::cout << "ERROR: vertex not in wanted format in OBJLoader" << "\n";
						system("PAUSE");
						exit(-1);
					}
				}

				else if (buffer[0] == 'f' && (buffer[1] == ' ' || buffer[1] == 32))  /// faces, 20 in icosahedron
				{

					int numNodes = 0;

					std::stringstream stream(buffer);

					// assign the first triangle:
					int v1, v2, v3, v_extra;
					char dummy;
					stream >> dummy >> v1 >> v2 >> v3;
					mesh.faces.push_back(Vec3i(v1, v2, v3));

					while (stream >> v_extra) {
						v2 = v3;
						v3 = v_extra;
						mesh.faces.push_back(Vec3i(v1, v2, v3));

						//	std::cout << v1 << " " << v2 << " " << v3 << std::endl;
					}
					*/

					// from here
					
					std::ifstream ifs(filenamestring.c_str(), std::ifstream::in);

					if (!ifs.good())
					{
					std::cout << "ERROR: loading obj:(" << filename << ") file not found or not good" << "\n";
					system("PAUSE");
					exit(0);
					}


					std::string line, key;
					while (!ifs.eof() && std::getline(ifs, line)) {
					key = "";
					std::stringstream stringstream(line);
					stringstream >> key >> std::ws;

					// if (sscanf(buffer, "v %f %f %f", &f1, &f2, &f3) == 3){
					// mesh.verts.push_back(Vec3f(f1, f2, f3));

					if (key == "v") { // vertex	
						float x, y, z;
						while (!stringstream.eof()) {
							stringstream >> x >> std::ws >> y >> std::ws >> z >> std::ws;
							mesh.verts.push_back(Vec3f(x, y, z));
						}
					}
					else if (key == "vp") { // parameter
						float x;
						// std::vector<float> tempparameters;
						while (!stringstream.eof()) {
							stringstream >> x >> std::ws;
							// tempparameters.push_back(x);
						}
						//parameters.push_back(tempparameters);
					}
					else if (key == "vt") { // texture coordinate
						float x;
						// std::vector<float> temptexcoords;
						while (!stringstream.eof()) {
							stringstream >> x >> std::ws;
							// temptexcoords.push_back(x);
						}
						//texcoords.push_back(temptexcoords);
					}
					else if (key == "vn") { // normal
						float x;
						// std::vector<float> tempnormals;
						while (!stringstream.eof()) {
							stringstream >> x >> std::ws;
							//	tempnormals.push_back(x);
						}
						//tempnormal.normalize();
						//normals.push_back(tempnormals);
					}

					else if (key == "f") { // face
						face f;
						int v, t, n;
						while (!stringstream.eof()) {
							stringstream >> v >> std::ws;
							f.vertex.push_back(v); // v - 1
							if (stringstream.peek() == '/') {
								stringstream.get();
								if (stringstream.peek() == '/') {
									stringstream.get();
									stringstream >> n >> std::ws;
									f.normal.push_back(n - 1);
								}
								else {
									stringstream >> t >> std::ws;
									f.texture.push_back(t - 1);
									if (stringstream.peek() == '/') {
										stringstream.get();
										stringstream >> n >> std::ws;
										f.normal.push_back(n - 1);
									}
								}
							}
						}

					int numtriangles = f.vertex.size() - 2; // 1 triangle if 3 vertices, 2 if 4 etc

					for (int i = 0; i < numtriangles; i++){  // first vertex remains the same for all triangles in a triangle fan
					mesh.faces.push_back(Vec3i(f.vertex[0], f.vertex[i + 1], f.vertex[i + 2]));
					}

					//while (stream >> v_extra) {
					//	v2 = v3;
					//	v3 = v_extra;
					//	mesh.faces.push_back(Vec3i(v1, v2, v3));
					//}
					}
					else {
					}
					

					/*
					char buffer[256], str[255];
					float f1, f2, f3;

					while (!in.getline(buffer, 255).eof()) // reads new line from file
					{
					buffer[255] = '\0'; // denotes end of stringbuffer
					sscanf_s(buffer, "%s", str, 255);

					// reading a vertex
					if (buffer[0] == 'v' && (buffer[1] == ' ' || buffer[1] == 32)){   /// vertices
					if (sscanf(buffer, "v %f %f %f", &f1, &f2, &f3) == 3){
					mesh.verts.push_back(Vec3f(f1, f2, f3));
					}
					else{
					std::cout << "ERROR: vertex not in wanted format in OBJLoader" << "\n";
					system("PAUSE");
					exit(-1);
					}
					}


					//while (std::getline(fin, line)) //Read a line
					////{
					//	std::stringstream ss(buffer);

					//	while (ss >> i) //Extract integers from line
					//		vec.push_back(i);

					//	V.push_back(vec);
					//	vec.clear();

					//}

					// reading faceMtls

					else if (buffer[0] == 'f' && (buffer[1] == ' ' || buffer[1] == 32))  /// faces
					{
					/////////////////
					/*

					int numNodes = 0;
					std::stringstream stream(buffer);

					face f;
					int v;
					int t;
					int n;
					while (!stream.eof()) {

					stream >> v >> std::ws;
					f.vertex.push_back(v - 1);
					if (stream.peek() == '/') {
					stream.get();
					if (stream.peek() == '/') {
					stream.get();
					stream >> n >> std::ws;
					f.normal.push_back(n - 1);
					}
					else {
					stream >> t >> std::ws;
					f.texture.push_back(t - 1);
					if (stream.peek() == '/') {
					stream.get();
					stream >> n >> std::ws;
					f.normal.push_back(n - 1);
					}
					}
					}
					} // end of while loop

					//faces.push_back(currface);
					if (currface.vertex.size() == 3)
					mesh.faces.push_back(Vec3i(currface.vertex[0], currface.vertex[1], currface.vertex[2]));
					*/
					///////////////////////////////////
					/*
										int numNodes = 0;
										std::stringstream stream(buffer);

										// assign the first triangle:
										int v1, v2, v3, v_extra;
										char dummy;
										stream >> dummy >> v1 >> v2 >> v3;
										mesh.faces.push_back(Vec3i(v1, v2, v3));

										while (stream >> v_extra) {
										v2 = v3;
										v3 = v_extra;
										mesh.faces.push_back(Vec3i(v1, v2, v3));

										//	std::cout << v1 << " " << v2 << " " << v3 << std::endl;
										}
										*/
					////////////////////////////////////////////
					/*
					face fc;
					int v, n, t;
					while (stream) {

					stream >> v >> std::ws;
					fc.vertex.push_back(v - 1);
					if (stream.peek() == '/') {
					stream.get();
					if (stream.peek() == '/') {
					stream.get();
					stream >> n >> std::ws;
					fc.normal.push_back(n - 1);
					}
					else {
					stream >> t >> std::ws;
					fc.texture.push_back(t - 1);
					if (stream.peek() == '/') {
					stream.get();
					stream >> n >> std::ws;
					fc.normal.push_back(n - 1);
					}
					}
					}
					} // end of while loop

					mesh.faces.push_back(Vec3i(fc.vertex[0], fc.vertex[1], fc.vertex[2]));
					*/

					/////////////////////////			

					//std::cout << "Done!" << std::endl;
					//while (my_stream >> dummy) std::cout << dummy << " ";

					//my_stream >> dummy >> v1 >> v2 >> v3;

					//mesh.faces.push_back(Vec3i(v1, v2, v3));

					// Check for more vertices on the same line and convert to triangles:

					/*
					do {

					mesh.faces.push_back(Vec3i(v1, v2, v3));
					} while (stream >> v2 >> v3);*/

					/*

					do {
					numNodes++;
					} while (my_stream >> word);

					numNodes=numNodes-2; // removing first character on the line: f
					//std::cout << numNodes << std::endl;

					Vec3i f;
					int nt;
					switch (numNodes)
					{
					case 3:

					nt = sscanf(buffer, "f %d %d %d", &f._v[0], &f._v[1], &f._v[2]);
					mesh.faces.push_back(f);
					break;
					case 4:
					int v1, v2, v3, v4;

					nt = sscanf(buffer, "f %d %d %d %d", &v1, &v2, &v3, &v4);

					f._v[0] = v1;
					f._v[1] = v2;
					f._v[2] = v3;
					mesh.faces.push_back(f);

					f._v[0] = v3;
					f._v[1] = v4;
					f._v[2] = v1;


					mesh.faces.push_back(f);
					break;

					default:
					break;
					}
					*/

					//Vec3i f;
					//int nt = sscanf(buffer, "f %d %d %d", &f._v[0], &f._v[1], &f._v[2]);
					//if (nt != 3){
					//	std::cout << "ERROR: Don't recognize face format (vertex indices) " << "\n";
					//	system("PAUSE");
					//	exit(-1);
					//}
					//mesh.faces.push_back(f);
					//			}
			//	}
			} // end of while loop

			totalVertices = mesh.verts.size();
			totalTriangles = mesh.faces.size();

			vertices = (Vertex *)malloc(totalVertices*sizeof(Vertex));
			verticesNo = totalVertices;
			pCurrentVertex = vertices;

			triangles = (Triangle *)malloc(totalTriangles*sizeof(Triangle));
			trianglesNo = totalTriangles;
			pCurrentTriangle = triangles;

			std::cout << "total vertices: " << totalVertices << "\n";
			std::cout << "total triangles: " << totalTriangles << "\n";

			for (int i = 0; i < totalVertices; i++){
				Vec3f currentvert = mesh.verts[i];
				pCurrentVertex->x = currentvert.x;
				pCurrentVertex->y = currentvert.y;
				pCurrentVertex->z = currentvert.z;
				pCurrentVertex->_normal.x = 0.f; // not used for now, normals are computed on-the-fly by GPU
				pCurrentVertex->_normal.y = 0.f; // not used 
				pCurrentVertex->_normal.z = 0.f; // not used 

				pCurrentVertex++;
			}

			std::cout << "Vertices loaded\n";

//			for (int i = 0; i < totalTriangles; i++)
			while(totalTriangles)
			{ 
				totalTriangles--;
				
				Vec3i currentfaceinds = mesh.faces[totalTriangles];

				pCurrentTriangle->_idx1 = currentfaceinds.x - 1;
				pCurrentTriangle->_idx2 = currentfaceinds.y - 1;
				pCurrentTriangle->_idx3 = currentfaceinds.z - 1;

				Vertex *vertexA = &vertices[currentfaceinds.x - 1];
				Vertex *vertexB = &vertices[currentfaceinds.y - 1];
				Vertex *vertexC = &vertices[currentfaceinds.z - 1];

				pCurrentTriangle++;
			}
		}

		else
			panic("Unknown extension (only .ply and .obj accepted)");
	}
	else
		panic("No extension in filename (only .ply accepted)");

	std::cout << "Vertices:  " << verticesNo << std::endl;
	std::cout << "Triangles: " << trianglesNo << std::endl;

}

	/////////////////////////
	// SCENE GEOMETRY PROCESSING
	/////////////////////////

float processgeo(){

	// Center scene at world's center

	Vec3f minp(FLT_MAX, FLT_MAX, FLT_MAX);
	Vec3f maxp(-FLT_MAX, -FLT_MAX, -FLT_MAX);

	// calculate min and max bounds of scene
	// loop over all triangles in scene, grow minp and maxp
	for (unsigned i = 0; i<trianglesNo; i++) {

		minp = min3f(minp, vertices[triangles[i]._idx1]);
		minp = min3f(minp, vertices[triangles[i]._idx2]);
		minp = min3f(minp, vertices[triangles[i]._idx3]);

		maxp = max3f(maxp, vertices[triangles[i]._idx1]);
		maxp = max3f(maxp, vertices[triangles[i]._idx2]);
		maxp = max3f(maxp, vertices[triangles[i]._idx3]);
	}

	// scene bounding box center before scaling and translating
	Vec3f origCenter = Vec3f(
		(maxp.x + minp.x) * 0.5,
		(maxp.y + minp.y) * 0.5,
		(maxp.z + minp.z) * 0.5);

	minp -= origCenter;
	maxp -= origCenter;

	// Scale scene so max(abs x,y,z coordinates) = MaxCoordAfterRescale

	float maxi = 0;
	maxi = std::max(maxi, (float)fabs(minp.x));
	maxi = std::max(maxi, (float)fabs(minp.y));
	maxi = std::max(maxi, (float)fabs(minp.z));
	maxi = std::max(maxi, (float)fabs(maxp.x));
	maxi = std::max(maxi, (float)fabs(maxp.y));
	maxi = std::max(maxi, (float)fabs(maxp.z));

	std::cout << "Scaling factor: " << (MaxCoordAfterRescale / maxi) << "\n";
	std::cout << "Center origin: " << origCenter.x << " " << origCenter.y << " " << origCenter.z << "\n";

	std::cout << "\nCentering and scaling vertices..." << std::endl;
	for (unsigned i = 0; i<verticesNo; i++) {
		vertices[i] -= origCenter;
		//vertices[i].y += origCenter.y;
		//vertices[i] *= (MaxCoordAfterRescale / maxi);
		vertices[i] *= 0.1; // 0.25
	}

	return MaxCoordAfterRescale;
}