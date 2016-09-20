// code for depth-of-field, mouse + keyboard user interaction based on https://github.com/peterkutz/GPUPathTracer

#pragma once
#include "camera.h"

InteractiveCamera* interactiveCamera = NULL;

bool buffer_reset = false;

void initCamera()
{
	delete interactiveCamera;
	interactiveCamera = new InteractiveCamera();

	interactiveCamera->setResolution(scrwidth, scrheight);
	interactiveCamera->setFOVX(45);
}

// mouse event handlers
int lastX = 0, lastY = 0;
int theButtonState = 0;
int theModifierState = 0;

// keyboard interaction
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key) {

	case(27) : exit(0);
	case(' ') : initCamera(); buffer_reset = true; break;
	case('a') : interactiveCamera->strafe(-0.05f); buffer_reset = true; break;
	case('d') : interactiveCamera->strafe(0.05f); buffer_reset = true; break;
	case('r') : interactiveCamera->changeAltitude(0.05f); buffer_reset = true; break;
	case('f') : interactiveCamera->changeAltitude(-0.05f); buffer_reset = true; break;
	case('w') : interactiveCamera->goForward(0.05f); buffer_reset = true; break;
	case('s') : interactiveCamera->goForward(-0.05f); buffer_reset = true; break;
	case('g') : interactiveCamera->changeApertureDiameter(0.1); buffer_reset = true; break;
	case('h') : interactiveCamera->changeApertureDiameter(-0.1); buffer_reset = true; break;
	case('t') : interactiveCamera->changeFocalDistance(0.1); buffer_reset = true; break;
	case('y') : interactiveCamera->changeFocalDistance(-0.1); buffer_reset = true; break;
	}
}

void specialkeys(int key, int, int){

	switch (key) {

	case GLUT_KEY_LEFT: interactiveCamera->changeYaw(0.02f); buffer_reset = true; break;
	case GLUT_KEY_RIGHT: interactiveCamera->changeYaw(-0.02f); buffer_reset = true; break;
	case GLUT_KEY_UP: interactiveCamera->changePitch(0.02f); buffer_reset = true; break;
	case GLUT_KEY_DOWN: interactiveCamera->changePitch(-0.02f); buffer_reset = true; break;

	}
}

// camera mouse controls in X and Y direction
void motion(int x, int y)
{
	int deltaX = lastX - x;
	int deltaY = lastY - y;

	if (deltaX != 0 || deltaY != 0) {

		if (theButtonState == GLUT_LEFT_BUTTON)  // Rotate
		{
			interactiveCamera->changeYaw(deltaX * 0.01);
			interactiveCamera->changePitch(-deltaY * 0.01);
		}
		else if (theButtonState == GLUT_MIDDLE_BUTTON) // Zoom
		{
			interactiveCamera->changeAltitude(-deltaY * 0.01);
		}

		if (theButtonState == GLUT_RIGHT_BUTTON) // camera move
		{
			interactiveCamera->changeRadius(-deltaY * 0.01);
		}

		lastX = x;
		lastY = y;
		buffer_reset = true;
		glutPostRedisplay();

	}
}

void mouse(int button, int state, int x, int y)
{
	theButtonState = button;
	theModifierState = glutGetModifiers();
	lastX = x;
	lastY = y;
	motion(x, y);
}