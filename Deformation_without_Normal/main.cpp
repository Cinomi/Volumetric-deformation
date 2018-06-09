#include <igl/readOFF.h>
#include <igl/viewer/Viewer.h>
#include <igl/per_vertex_normals.h>
#include <igl/per_face_normals.h>

#include "nanoflann.hpp"
#include "nanogui/formhelper.h"
#include "nanogui/screen.h"

using namespace Eigen;
using namespace std;



// ****************************************************** //
// ***************** Global Coefficients **************** //
// ****************************************************** //

MatrixXd V;
MatrixXi F;
MatrixXd N;

// Tapering
double taperingAlpha0 = 1.0;
double taperingAlpha1 = 0.3;

// Twisting
double twistScale = 90;

// Bending
double centerOfBend = 0.0;
double yMax = 0.1;
double yMin = -0.1;
double bendRate = 0.2;

// Initialization
Eigen::Matrix3d axisStart;
Eigen::Matrix3d axisEnd;
double axisScale = 0.1;
double normalsScale = 0.01;
bool showNormals = false;
igl::viewer::Viewer viewer;




// ****************************************************** //
// ********************* FUNCTIONS ********************** //
// ****************************************************** //

// ******************************************************
// Forward and Inverse Tapering
// ******************************************************
void linearTaperZ(MatrixXd& V, double a0, double a1) {
    for (int i = 0; i < V.rows(); ++i) {
        double r = a0+a1*V(i,2);
        double rPrim = a1;
        double x = V(i,0);
        double y = V(i,1);
        
        V(i,0) = r*x;
        V(i,1) = r*y;
    }
}

void inverseLinearTaperZ(MatrixXd& V, double a0, double a1) {
    for (int i = 0; i < V.rows(); ++i) {
        double r = a0+a1*V(i,2);
        double rPrim = a1;
        double X = V(i,0);
        double Y = V(i,1);
        
        V(i,0) = X/r;
        V(i,1) = Y/r;
    }
}



// ******************************************************
// Forward and Inverse Twisting
// ******************************************************
void twistModelZ(MatrixXd &V, const double twistScale) {
    const double degreesToRadians = 0.0174532925;
    for (int i=0; i<V.rows(); i++) {
        double z = V(i,2);
        double theta = degreesToRadians*twistScale*z;
        double C_theta = cos(theta);
        double S_theta = sin(theta);
        
        // Get old values of x nad y
        double x = V(i,0);
        double y = V(i,1);
        
        // Calculate new coordinates
        V(i,0) = x*C_theta-y*S_theta;
        V(i,1) = x*S_theta+y*C_theta;
    }
}

void inverseTwistModelZ(MatrixXd &V, const double twistScale) {
    const double degreesToRadians = 0.0174532925;
    for (int i=0; i<V.rows(); i++) {
        double Z = V(i,2);
        double theta = degreesToRadians*twistScale*Z;
        double C_theta = cos(theta);
        double S_theta = sin(theta);
        
        double X = V(i,0);
        double Y = V(i,1);
        
        V(i,0) = X*C_theta+Y*S_theta;
        V(i,1) = -X*S_theta+Y*C_theta;
    }
}


// ******************************************************
// Bending
// ******************************************************
void bendModel(MatrixXd &V, double y0, double yMax, double yMin, double k) {
    
    for (int i = 0; i < V.rows(); ++i) {
        // Collect current coordinates
        double x = V(i,0);
        double y = V(i,1);
        double z = V(i,2);
        
        // Pick y-hat
        double yHat = y;
        if (yHat<yMin){
            yHat = yMin;
        }else if (yHat>yMax){
            yHat = yMax;
        }
        
        // Bending angle
        double theta = k*(yHat-y0);
        double C_theta = cos(theta);
        double S_theta = sin(theta);
        
        // Calculate Y
        if (y>=yMin && y<=yMax) {
            V(i,1) = -S_theta*(z-(1.0/k))+y0;
        } else if (y<yMin) {
            V(i,1) = -S_theta*(z-(1.0/k))+y0+C_theta*(y-yMin);
        } else if (y>yMax) {
            V(i,1) = -S_theta*(z-(1.0/k))+y0+C_theta*(y-yMax);
        }
        
        // Calculate Z
        if (y>=yMin && y<=yMax) {
            V(i,2) = C_theta*(z-(1.0/k))+(1.0/k);
        } else if (y<yMin) {
            V(i,2) = C_theta*(z-(1.0/k))+(1.0/k)+S_theta*(y-yMin);
        } else if (y>yMax) {
            V(i,2) = C_theta*(z-(1.0/k))+(1.0/k)+S_theta*(y-yMax);
        }
        
        // Calculate k-hat
        double khat = (y == yHat) ? k : 0.0;
    }
    
}




// **************************************************** //
// ********************** MAIN ************************ //
// **************************************************** //
int main(int argc, char *argv[]) {
    
    igl::readOFF(TUTORIAL_SHARED_PATH "/bunny.off", V, F);
    
    // Move mesh to coords center
    MatrixXd centerOfMass = V.colwise().sum();
    centerOfMass /= V.rows();
    for (int i = 0; i < V.rows(); ++i){
        V.row(i) -= centerOfMass;
    }
    
    // Visualize the mesh in a viewer
    igl::viewer::Viewer viewer;
    {
        viewer.core.show_lines = false;
        viewer.core.show_overlay_depth = false;
    }
    
    // Viewer menu
    viewer.callback_init = [&](igl::viewer::Viewer &viewer) {
        viewer.ngui->addWindow(Eigen::Vector2i(900, 10), "Transformations");
        
        //***********************************************************************
        // Reset
        //***********************************************************************
        viewer.ngui->addGroup("Reset");
        viewer.ngui->addButton("Reset Mesh", [&]() {
            igl::readOFF(TUTORIAL_SHARED_PATH "/bunny.off", V, F);
            
            // move mesh to coords center
            MatrixXd centerOfMass = V.colwise().sum();
            centerOfMass /= V.rows();
            for (int i = 0; i < V.rows(); ++i) V.row(i) -= centerOfMass;
            
            igl::per_vertex_normals(V,F,N);
            viewer.data.clear();
            viewer.data.set_mesh(V,F);
        });
        
        //***********************************************************************
        // Tapering
        //***********************************************************************
        viewer.ngui->addGroup("Tapering");
        viewer.ngui->addVariable<double>("Tapering Function (alpha0)",[&](double val){
                                            taperingAlpha0 = val;
                                        },[&](){
                                            return taperingAlpha0;
                                        });
        //***********************************************************************
        viewer.ngui->addVariable<double>("Tapering Function (alpha1)",[&](double val){
                                            taperingAlpha1 = val;
                                        },[&](){
                                            return taperingAlpha1;
                                        });
        //***********************************************************************
        viewer.ngui->addButton("Taper", [&]() {
            linearTaperZ(V,taperingAlpha0,taperingAlpha1);
            viewer.data.clear();
            viewer.data.set_mesh(V,F);
        });
        //***********************************************************************
        viewer.ngui->addButton("Inverse Taper", [&]() {
            inverseLinearTaperZ(V,taperingAlpha0,taperingAlpha1);
            viewer.data.clear();
            viewer.data.set_mesh(V,F);
        });
        
        
        
        
        //***********************************************************************
        // Twisting
        //***********************************************************************
        viewer.ngui->addGroup("Twisting");
        viewer.ngui->addVariable<double>("Strength along Z-axis", [&](double val) {
                                            twistScale = val;
                                        }, [&]() {
                                            return twistScale;
                                        });
        //**************************************************************************
        viewer.ngui->addButton("Twist", [&]() {
            twistModelZ(V,twistScale);
            viewer.data.clear();
            viewer.data.set_mesh(V,F);
        });
        //**************************************************************************
        viewer.ngui->addButton("Inverse Twist", [&]() {
            inverseTwistModelZ(V,twistScale);
            viewer.data.clear();
            viewer.data.set_mesh(V,F);
        });
        
        
        
        //***********************************************************************
        // Bending
        //***********************************************************************
        viewer.ngui->addGroup("Bending");
        viewer.ngui->addVariable<double>("Center of bend", [&](double val){
                                            centerOfBend = val;
                                        }, [&](){
                                            return centerOfBend;
                                        });
        //**************************************************************************
        viewer.ngui->addVariable<double>("Y min", [&](double val){
                                            yMin = val;
                                        }, [&](){
                                            return yMin;
                                        });
        //**************************************************************************
        viewer.ngui->addVariable<double>("Y max", [&](double val){
                                            yMax = val;
                                        }, [&](){
                                            return yMax;
                                        });
        //**************************************************************************
        viewer.ngui->addVariable<double>("Bend rate", [&](double val){
                                            bendRate = val;
                                        }, [&](){
                                            return bendRate;
                                        });
        //**************************************************************************
        viewer.ngui->addButton("Bend", [&]() {
            bendModel(V,centerOfBend,yMax,yMin,bendRate);
            viewer.data.clear();
            viewer.data.set_mesh(V,F);
        });

        
        viewer.screen->performLayout();
        return false;
    };
    
    viewer.data.clear();
    viewer.data.set_mesh(V,F);
    
    viewer.launch();
    return 0;
}
