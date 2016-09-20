#include "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\extras\CUPTI\include\GL\glew.h"
#include "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\extras\CUPTI\include\GL\glut.h"
#include <sstream>
#include <iostream>
#include "SceneLoader.h"
#include "Camera.h"
#include "Array.h"
#include "Scene.h"
#include "Util.h"
#include "BVH.h"
#include "CudaBVH.h"
#include "CudaRenderKernel.h"
#include "HDRloader.h"
#include "MouseKeyboardInput.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#ifndef M_PI
#define M_PI 3.14156265
#endif

// test scenes

//const char* scenefile = "data/icosahedron.obj";
//const char* scenefile = "data/dragon_vrip_res3.ply";  
const char* scenefile = "data/dragon_vrip.ply"; 
//const char* scenefile = "data/happy_vrip_res2.ply";  
//const char* scenefile = "data/happy_vrip.ply"; 
//const char* scenefile = "data/bun_zipper.ply";  
//const char* scenefile = "data/trumpet.obj";      // minicooper.obj, cessna.obj
//const char* scenefile = "data/italianfromblender2.obj"; 
//const char* scenefile = "data/dragon.obj"; 
//const char* scenefile = "data/sponza_crytek.obj"; 

// HDR environment

//const char* HDRmapname = "data/ArboretumInBloom_Ref.hdr"; 
const char* HDRmapname = "data/Topanga_Forest_B_3k.hdr";
//const char* HDRmapname = "data/Ditch-River_2k.hdr";
//const char* HDRmapname = "data/GCanyon_C_YumaPoint_3k.hdr";

Vec4i* cpuNodePtr = NULL;
Vec4i* cpuTriWoopPtr = NULL;
Vec4i* cpuTriDebugPtr = NULL;
Vec4f* cpuTriNormalPtr = NULL;
S32*   cpuTriIndicesPtr = NULL;

float4* cudaNodePtr = NULL;
float4* cudaTriWoopPtr = NULL;
float4* cudaTriDebugPtr = NULL;
float4* cudaTriNormalPtr = NULL;
S32*    cudaTriIndicesPtr = NULL;

Camera* cudaRendercam = NULL;
Camera* hostRendercam = NULL;
Vec3f* accumulatebuffer = NULL; // image buffer storing accumulated pixel samples
Vec3f* finaloutputbuffer = NULL; // stores averaged pixel samples
float4* gpuHDRenv = NULL;
Vec4f* cpuHDRenv = NULL;
Vec4f* m_triNormals = NULL;
CudaBVH* gpuBVH = NULL;

Clock watch;
GLuint vbo;

int framenumber = 0;
int nodeSize = 0;
int leafnode_count = 0;
int triangle_count = 0;
int triWoopSize = 0;
int triDebugSize = 0;
int triIndicesSize = 0;
float scalefactor = 1.2f;
__device__ float timer = 0.0f;
bool nocachedBVH = false;


void Timer(int obsolete) {
	glutPostRedisplay();
	glutTimerFunc(30, Timer, 0);
}

void createVBO(GLuint* vbo)
{
	//Create vertex buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	//Initialize VBO
	unsigned int size = scrwidth * scrheight * sizeof(Vec3f);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//Register VBO with CUDA
	cudaGLRegisterBufferObject(*vbo);
}

// display function called by glutMainLoop(), gets executed every frame 
void disp(void)
{
	// if camera has moved, reset the accumulation buffer
	if (buffer_reset){ cudaMemset(accumulatebuffer, 1, scrwidth * scrheight * sizeof(Vec3f)); framenumber = 0; }

	buffer_reset = false;
	framenumber++;

	// build a new camera for each frame on the CPU
	interactiveCamera->buildRenderCamera(hostRendercam);

	// copy the CPU camera to a GPU camera
	cudaMemcpy(cudaRendercam, hostRendercam, sizeof(Camera), cudaMemcpyHostToDevice);

	cudaThreadSynchronize();
	cudaGLMapBufferObject((void**)&finaloutputbuffer, vbo); // maps a buffer object for access by CUDA

	glClear(GL_COLOR_BUFFER_BIT); //clear all pixels

	// calculate a new seed for the random number generator, based on the framenumber
	unsigned int hashedframes = WangHash(framenumber);

	// gateway from host to CUDA, passes all data needed to render frame (triangles, BVH tree, camera) to CUDA for execution
	cudaRender(cudaNodePtr, cudaTriWoopPtr, cudaTriDebugPtr, cudaTriIndicesPtr, finaloutputbuffer,
		accumulatebuffer, gpuHDRenv, framenumber, hashedframes, nodeSize, leafnode_count, triangle_count, cudaRendercam);
	
	cudaThreadSynchronize();
	cudaGLUnmapBufferObject(vbo);
	//glFlush();
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(2, GL_FLOAT, 12, 0);
	glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid*)8);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_POINTS, 0, scrwidth * scrheight);
	glDisableClientState(GL_VERTEX_ARRAY);

	glutSwapBuffers();
}

void loadBVHfromCache(FILE* BVHcachefile, const std::string BVHcacheFilename)
{
	if (1 != fread(&nodeSize, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error reading BVH cache file!\n";
	if (1 != fread(&triangle_count, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error reading BVH cache file!\n";
	if (1 != fread(&leafnode_count, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error reading BVH cache file!\n";
	if (1 != fread(&triWoopSize, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error reading BVH cache file!\n";
	if (1 != fread(&triDebugSize, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error reading BVH cache file!\n";
	if (1 != fread(&triIndicesSize, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error reading BVH cache file!\n";

	std::cout << "Number of nodes: " << nodeSize << "\n";
	std::cout << "Number of triangles: " << triangle_count << "\n";
	std::cout << "Number of BVH leafnodes: " << leafnode_count << "\n";

	cpuNodePtr = (Vec4i*)malloc(nodeSize * sizeof(Vec4i));
	cpuTriWoopPtr = (Vec4i*)malloc(triWoopSize * sizeof(Vec4i));
	cpuTriDebugPtr = (Vec4i*)malloc(triDebugSize * sizeof(Vec4i));
	cpuTriIndicesPtr = (S32*)malloc(triIndicesSize * sizeof(S32));

	if (nodeSize != fread(cpuNodePtr, sizeof(Vec4i), nodeSize, BVHcachefile)) std::cout << "Error reading BVH cache file!\n";
	if (triWoopSize != fread(cpuTriWoopPtr, sizeof(Vec4i), triWoopSize, BVHcachefile)) std::cout << "Error reading BVH cache file!\n";
	if (triDebugSize != fread(cpuTriDebugPtr, sizeof(Vec4i), triDebugSize, BVHcachefile)) std::cout << "Error reading BVH cache file!\n";
	if (triIndicesSize != fread(cpuTriIndicesPtr, sizeof(S32), triIndicesSize, BVHcachefile)) std::cout << "Error reading BVH cache file!\n";

	fclose(BVHcachefile);
	std::cout << "Successfully loaded BVH from cache file!\n";
}

void writeBVHcachefile(FILE* BVHcachefile, const std::string BVHcacheFilename){

	BVHcachefile = fopen(BVHcacheFilename.c_str(), "wb");
	if (!BVHcachefile) std::cout << "Error opening BVH cache file!\n";
	if (1 != fwrite(&nodeSize, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error writing BVH cache file!\n";
	if (1 != fwrite(&triangle_count, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error writing BVH cache file!\n";
	if (1 != fwrite(&leafnode_count, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error writing BVH cache file!\n";
	if (1 != fwrite(&triWoopSize, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error writing BVH cache file!\n";
	if (1 != fwrite(&triDebugSize, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error writing BVH cache file!\n";
	if (1 != fwrite(&triIndicesSize, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error writing BVH cache file!\n";
	if (nodeSize != fwrite(cpuNodePtr, sizeof(Vec4i), nodeSize, BVHcachefile)) std::cout << "Error writing BVH cache file!\n";
	if (triWoopSize != fwrite(cpuTriWoopPtr, sizeof(Vec4i), triWoopSize, BVHcachefile)) std::cout << "Error writing BVH cache file!\n";
	if (triDebugSize != fwrite(cpuTriDebugPtr, sizeof(Vec4i), triDebugSize, BVHcachefile)) std::cout << "Error writing BVH cache file!\n";
	if (triIndicesSize != fwrite(cpuTriIndicesPtr, sizeof(S32), triIndicesSize, BVHcachefile)) std::cout << "Error writing BVH cache file!\n";
	
	fclose(BVHcachefile);
	std::cout << "Successfully created BVH cache file!\n";
}

// initialise HDR environment map
// from https://graphics.stanford.edu/wikis/cs148-11-summer/HDRIlluminator

void initHDR(){
	
	HDRImage HDRresult;
	const char* HDRfile = HDRmapname;

	if (HDRLoader::load(HDRfile, HDRresult)) 
		printf("HDR environment map loaded. Width: %d Height: %d\n", HDRresult.width, HDRresult.height);
	else printf("HDR environment map not found\n");

	int HDRwidth = HDRresult.width;
	int HDRheight = HDRresult.height;
	cpuHDRenv = new Vec4f[HDRwidth * HDRheight];
	//_data = new RGBColor[width*height];

	for (int i = 0; i<HDRwidth; i++){
		for (int j = 0; j<HDRheight; j++){
			int idx = 3 * (HDRwidth*j + i);
			//int idx2 = width*(height-j-1)+i;
			int idx2 = HDRwidth*(j)+i;
			cpuHDRenv[idx2] = Vec4f(HDRresult.colors[idx], HDRresult.colors[idx + 1], HDRresult.colors[idx + 2], 0.0f);
		}
	}

	// copy HDR map to CUDA
	cudaMalloc(&gpuHDRenv, HDRwidth * HDRheight * sizeof(float4));
	cudaMemcpy(gpuHDRenv, cpuHDRenv, HDRwidth * HDRheight * sizeof(float4), cudaMemcpyHostToDevice);
}

void initCUDAscenedata(){

	// allocate GPU memory for accumulation buffer
	cudaMalloc(&accumulatebuffer, scrwidth * scrheight * sizeof(Vec3f));
	
	// allocate GPU memory for interactive camera
	cudaMalloc((void**)&cudaRendercam, sizeof(Camera));

	// allocate and copy scene databuffers to the GPU (BVH nodes, triangle vertices, triangle indices)
	cudaMalloc((void**)&cudaNodePtr, nodeSize * sizeof(float4));
	cudaMemcpy(cudaNodePtr, cpuNodePtr, nodeSize * sizeof(float4), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&cudaTriWoopPtr, triWoopSize * sizeof(float4));
	cudaMemcpy(cudaTriWoopPtr, cpuTriWoopPtr, triWoopSize * sizeof(float4), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&cudaTriDebugPtr, triDebugSize * sizeof(float4));
	cudaMemcpy(cudaTriDebugPtr, cpuTriDebugPtr, triDebugSize * sizeof(float4), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&cudaTriIndicesPtr, triIndicesSize * sizeof(S32));
	cudaMemcpy(cudaTriIndicesPtr, cpuTriIndicesPtr, triIndicesSize * sizeof(S32), cudaMemcpyHostToDevice);

	std::cout << "Scene data copied to CUDA\n";
}

void createBVH(){
	
	load_object(scenefile);
	float maxi2 = processgeo();

	std::cout << "Scene geometry loaded and processed\n";

	// create arrays for the triangles and the vertices
	// Scene() constructor: Scene(const S32 numTris, const S32 numVerts, const Array<Triangle>& tris, const Array<Vec3f>& verts)

	Array<Scene::Triangle> tris;
	Array<Vec3f> verts;
	tris.clear();
	verts.clear();

	// convert Triangle to Scene::Triangle
	for (unsigned int i = 0; i < trianglesNo; i++){
		Scene::Triangle newtri;
		newtri.vertices = Vec3i(triangles[i]._idx1, triangles[i]._idx2, triangles[i]._idx3);
		tris.add(newtri);
	}

	// fill up Array of vertices
	for (unsigned int i = 0; i < verticesNo; i++) { 
		verts.add(Vec3f(vertices[i].x, vertices[i].y, vertices[i].z)); 
	}

	std::cout << "Building a new scene\n";
	Scene* scene = new Scene(trianglesNo, verticesNo, tris, verts);

	std::cout << "Building BVH with spatial splits\n";
	// create a default platform
	Platform defaultplatform;
	BVH::BuildParams defaultparams;
	BVH::Stats stats;
	BVH myBVH(scene, defaultplatform, defaultparams);

	std::cout << "Building CudaBVH\n";
	// create CUDA friendly BVH datastructure
	gpuBVH = new CudaBVH(myBVH, BVHLayout_Compact);  // Fermi BVH layout = compact. BVH layout for Kepler kernel Compact2
	std::cout << "CudaBVH successfully created\n";

	std::cout << "Hi Sam!  How you doin'?" << std::endl;

	cpuNodePtr = gpuBVH->getGpuNodes();
	cpuTriWoopPtr = gpuBVH->getGpuTriWoop();
	cpuTriDebugPtr = gpuBVH->getDebugTri();
	cpuTriIndicesPtr = gpuBVH->getGpuTriIndices();
	cpuTriNormalPtr = m_triNormals;

	nodeSize = gpuBVH->getGpuNodesSize();
	triWoopSize = gpuBVH->getGpuTriWoopSize();
	triDebugSize = gpuBVH->getDebugTriSize();
	triIndicesSize = gpuBVH->getGpuTriIndicesSize();
	leafnode_count = gpuBVH->getLeafnodeCount();
	triangle_count = gpuBVH->getTriCount();
}

void deleteCudaAndCpuMemory(){
	// free CUDA memory
	cudaFree(cudaNodePtr);
	cudaFree(cudaTriWoopPtr);
	cudaFree(cudaTriDebugPtr);
	cudaFree(cudaTriNormalPtr);
	cudaFree(cudaTriIndicesPtr);
	cudaFree(cudaRendercam);
	cudaFree(accumulatebuffer);
	cudaFree(finaloutputbuffer);
	cudaFree(gpuHDRenv);

	// release CPU memory
	free(cpuNodePtr);
	free(cpuTriWoopPtr);
	free(cpuTriDebugPtr);
	free(cpuTriNormalPtr);
	free(cpuTriIndicesPtr);

	delete hostRendercam;
	delete interactiveCamera;
	delete cpuHDRenv;
	delete gpuBVH;
}


int main(int argc, char** argv){

	// create a CPU camera
	hostRendercam = new Camera();
	// initialise an interactive camera on the CPU side
	initCamera();	
	interactiveCamera->buildRenderCamera(hostRendercam);

	std::string BVHcacheFilename(scenefile);
	BVHcacheFilename += ".bvh";  

	FILE* BVHcachefile = fopen(BVHcacheFilename.c_str(), "rb");
	if (!BVHcachefile){ nocachedBVH = true; }
	
	//if (true){ // overrule cache
	if (nocachedBVH){
		std::cout << "No cached BVH file available\nCreating new BVH...\n";
		// initialise all data needed to start rendering (BVH data, triangles, vertices)
		createBVH();
		// store the BVH in a file
		writeBVHcachefile(BVHcachefile, BVHcacheFilename);
	}

	else { // cached BVH available
		std::cout << "Cached BVH available\nReading " << BVHcacheFilename << "...\n";
		loadBVHfromCache(BVHcachefile, BVHcacheFilename); 
	}
	
	initCUDAscenedata(); // copy scene data to the GPU, ready to be used by CUDA
	initHDR(); // initialise the HDR environment map

	// initialise GLUT
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB); // specify the display mode to be RGB and single buffering
	glutInitWindowPosition(100, 100); // specify the initial window position
	glutInitWindowSize(scrwidth, scrheight); // specify the initial window size
	glutCreateWindow("MatchingSocks, CUDA path tracer using SplitBVH"); // create the window and set title

	// initialise OpenGL:
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0.0, scrwidth, 0.0, scrheight);
	fprintf(stderr, "OpenGL initialized \n");

	// register callback function to display graphics
	glutDisplayFunc(disp);

	// functions for user interaction
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(specialkeys);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);

	// initialise GLEW
	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 ")) {
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		exit(0);
	}
	fprintf(stderr, "glew initialized  \n");

	// call Timer()
	Timer(0);
	createVBO(&vbo);
	fprintf(stderr, "VBO created  \n");
	// enter the main loop and start rendering
	fprintf(stderr, "Entering glutMainLoop...  \n");
	printf("Rendering started...\n");
	glutMainLoop();

	deleteCudaAndCpuMemory();
}