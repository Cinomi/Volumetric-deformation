#include "acq/normalEstimation.h"
#include "acq/decoratedCloud.h"
#include "acq/cloudManager.h"

#include "nanogui/formhelper.h"
#include "nanogui/screen.h"

#include "igl/readOFF.h"
#include "igl/viewer/Viewer.h"
#include <igl/jet.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
#include <Eigen/IterativeLinearSolvers>
#include <igl/per_face_normals.h>

#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <unsupported/Eigen/MatrixFunctions>
#include <stdio.h>
#include <igl/edges.h>
#include <igl/jet.h>

using namespace std::chrono;
using namespace Eigen;
using namespace std;


// ****************************************************** //
// ********************* FUNCTIONS ********************** //
// ****************************************************** //

// ******************************************************
// Get Face Normal
// ******************************************************
MatrixXd FaceNormals(MatrixXd V, MatrixXi F) {
    MatrixXd FaceNormals(F.rows(),3);
    FaceNormals.setZero();
    for (int i=0;i<F.rows();i++) {
        RowVector3d v0,v1,v2;
        RowVector3d edge1,edge2;
        v0 = V.row(F(i,0));
        v1 = V.row(F(i,1));
        v2 = V.row(F(i,2));
        edge1 = v1-v0;
        edge2 = v2-v0;
        FaceNormals.row(i) = (edge1.cross(edge2)).normalized();
    }
    return FaceNormals;
}



// *******************************************************
//  Get Edges
// *******************************************************
MatrixXi GetEdges(MatrixXi faces, int nVertices) {
    MatrixXd edgeMatrix(nVertices, nVertices);
    edgeMatrix.setOnes();
    edgeMatrix = edgeMatrix * (-1);
    
    int nFaces = faces.rows();
    int numEdges = 0;
    for (int i = 0; i < nFaces; i++) {
        int v1 = faces(i, 0);
        int v2 = faces(i, 1);
        int v3 = faces(i, 2);
        
        if (edgeMatrix(v1, v2) == -1) {
            edgeMatrix(v1, v2) = 1;
            edgeMatrix(v2, v1) = 1;
            numEdges++;
            //std::cout << v1 << " " << v2 << std::endl;
        }
        if (edgeMatrix(v2, v3) == -1) {
            edgeMatrix(v2, v3) = 1;
            edgeMatrix(v3, v2) = 1;
            numEdges++;
            //std::cout << v2 << " " << v3 << std::endl;
        }
        if (edgeMatrix(v3, v1) == -1) {
            edgeMatrix(v3, v1) = 1;
            edgeMatrix(v1, v3) = 1;
            numEdges++;
            //std::cout << v3 << " " << v1 << std::endl;
        }
    }
    
    int edgeIndex = 0;
    MatrixXi edges(numEdges, 2);
    for (int i = 0; i < nVertices; i++) {
        for (int j = 0; j <= i; j++) {
            if (edgeMatrix(i, j) == 1) {
                edges(edgeIndex, 0) = i;
                edges(edgeIndex, 1) = j;
                edgeIndex++;
                //edgeMatrix(i, j) == -1;
                //edgeMatrix(j, i) == -1;
            }
        }
    }
    return edges;
}


// ****************************************************
//  Mean Coordinates
// ****************************************************
VectorXd MeanCoords(MatrixXd V_box, MatrixXi F_box, RowVector3d q) {
    MatrixXd unitVectors(V_box.rows(),3);
    unitVectors.setZero();
    VectorXd Distances(V_box.rows());
    Distances.setZero();
    VectorXd Weights(V_box.rows());
    Weights.setZero();
    
    // Mesh Point Check:
    bool abort = false;
    for (int j=0; j<V_box.rows(); j++) {
        if (q == V_box.row(j)) {
            Weights(j) = 1;
            abort = true;
        }
    }
    
    if (~abort) {
        double distThreshold = 0.0001;
        for (int k=0; k<V_box.rows(); k++) {
            Distances(k) = (V_box.row(k)-q).norm();
            unitVectors.row(k) = (V_box.row(k)-q) / Distances(k);
        }
        
        for (int j=0; j<F_box.rows(); j++) {
            double w1,w2,w3;
            Vector3d u1,u2,u3;
            u1 = (unitVectors.row(F_box(j,0))).transpose();
            u2 = (unitVectors.row(F_box(j,1))).transpose();
            u3 = (unitVectors.row(F_box(j,2))).transpose();
            
            double l1,l2,l3;
            l1 = (u2 - u3).norm();
            l2 = (u3 - u1).norm();
            l3 = (u1 - u2).norm();
            
            double theta1,theta2,theta3;
            theta1 = 2*asin(l1/2.0);
            theta2 = 2*asin(l2/2.0);
            theta3 = 2*asin(l3/2.0);
            
            double h;
            h = (theta1+theta2+theta3)/2.0;
            //If q lies inside the triangle plane, use simple triangle 2D interpolation and finish
            if ((M_PI - h)<distThreshold) {
                RowVector3d v1,v2,v3;
                v1 = V_box.row(F_box(j,0));
                v2 = V_box.row(F_box(j,1));
                v3 = V_box.row(F_box(j,2));
                
                double area1,area2,area3,totArea;
                area1 = 0.5*((v3-q).norm()*(v2-q).norm()*sin(acos((v3-q).dot(v2-q)/((v3-q).norm()*(v2-q).norm()))));
                area2 = 0.5*((v3-q).norm()*(v1-q).norm()*sin(acos((v3-q).dot(v1-q)/((v3-q).norm()*(v1-q).norm()))));
                area3 = 0.5*((v1-q).norm()*(v2-q).norm()*sin(acos((v1-q).dot(v2-q)/((v1-q).norm()*(v2-q).norm()))));
                totArea = 0.5*((v3-v1).norm()*(v2-v1).norm()*sin(acos((v3-v1).dot(v2-v1)/((v3-v1).norm()*(v2-v1).norm()))));
                w1 = area1/totArea;
                w2 = area2/totArea;
                w3 = area3/totArea;
                Weights.setZero();
                Weights(F_box(j,0)) = w1;
                Weights(F_box(j,1)) = w2;
                Weights(F_box(j,2)) = w3;
                break;
            }
            
            double c1,c2,c3;
            c1 = (2*sin(h)*sin(h-theta1)) / (sin(theta2)*sin(theta3)) - 1;
            c2 = (2*sin(h)*sin(h-theta2)) / (sin(theta3)*sin(theta1)) - 1;
            c3 = (2*sin(h)*sin(h-theta3)) / (sin(theta1)*sin(theta2)) - 1;
            MatrixXd tmpSign(3,3);
            tmpSign.setZero();
            tmpSign.col(0) = u1;
            tmpSign.col(1) = u2;
            tmpSign.col(2) = u3;
            
            double sign,s1,s2,s3;
            sign = copysign(1.0, tmpSign.determinant());
            s1 = sign*sqrt(1-c1*c1);
            s2 = sign*sqrt(1-c2*c2);
            s3 = sign*sqrt(1-c3*c3);
            
            //If q lies outside of the triangle but on the triangle plane, ignore the triangle
            if (!(abs(s1) <= 0.0001 || abs(s2) <= 0.0001 || abs(s3) <= 0.0001)) {
                w1 = (theta1 - c2*theta3 - c3*theta2) / (Distances(F_box(j,0))*sin(theta2)*s3);
                w2 = (theta2 - c3*theta1 - c1*theta3) / (Distances(F_box(j,1))*sin(theta3)*s1);
                w3 = (theta1 - c2*theta3 - c3*theta2) / (Distances(F_box(j,2))*sin(theta1)*s2);
                
                Weights(F_box(j,0)) = Weights(F_box(j,0))+w1;
                Weights(F_box(j,1)) = Weights(F_box(j,1))+w2;
                Weights(F_box(j,2)) = Weights(F_box(j,2))+w3;
            }
            
        }
        
    }
    Weights = Weights/Weights.sum();
    return Weights;
}



// ****************************************************
// Green Coordinates
// ****************************************************
double GCTriInt(RowVector3d p, RowVector3d v1, RowVector3d v2, RowVector3d eta) {
    double alpha,beta,lambda,c;
    alpha = acos((v2-v1).dot(p-v1)/(((v2-v1).norm())*((p-v1).norm())));
    beta = acos((v1-p).dot(v2-p)/(((v1-p).norm())*((v2-p).norm())));
    lambda = pow((p-v1).norm(),2) * pow(sin(alpha),2);
    c = pow((p-eta).norm(),2);
    
    double theta1,theta2;
    theta1 = M_PI - alpha;
    theta2 = M_PI - alpha - beta;
    
    double Itheta1,Itheta2;
    
    // for theta1
    double S1,C1;
    S1 = sin(theta1);
    C1 = cos(theta1);
    Itheta1 = (-copysign(1.0, S1)/2.0) *
    (2*sqrt(c)*atan((sqrt(c)*C1)/sqrt(lambda+S1*S1*c)) +
     sqrt(lambda)*log((2*sqrt(lambda)*S1*S1/pow(1-C1,2)*(1 - 2*c*C1/(c*(1+C1)+lambda+sqrt(lambda*lambda+lambda*c*S1*S1))))));
    
    // for theta2
    double S2,C2;
    S2 = sin(theta2);
    C2 = cos(theta2);
    Itheta2 = (-copysign(1.0, S2)/2.0) *
    (2*sqrt(c)*atan((sqrt(c)*C2)/sqrt(lambda+S2*S2*c)) +
     sqrt(lambda)*log((2*sqrt(lambda)*S2*S2/pow(1-C2,2)*(1 - 2*c*C2/(c*(1+C2)+lambda+sqrt(lambda*lambda+lambda*c*S2*S2))))));
    
    double value;
    value = (-1.0/(4*M_PI)) * abs(Itheta1-Itheta2-sqrt(c)*beta);
    
    return value;
}


tuple<RowVectorXd, RowVectorXd> GreenCoords(MatrixXd V_box, MatrixXi F_box, MatrixXd normals, RowVector3d q) {
    RowVectorXd phi(V_box.rows());
    phi.setZero();
    RowVectorXd psi(F_box.rows());
    psi.setZero();
    
    for (int j=0; j<F_box.rows(); j++) {
        RowVector3d Ntj,Vj1,Vj2,Vj3,p;
        Ntj = normals.row(j);
        Vj1 = V_box.row(F_box(j,0)) - q;
        Vj2 = V_box.row(F_box(j,1)) - q;
        Vj3 = V_box.row(F_box(j,2)) - q;
        p = Vj1.dot(Ntj)*Ntj;
        
        // l=1
        double s1,I1,II1;
        RowVector3d q1,N1;
        s1 = copysign(1.0, ((Vj1-p).cross(Vj2-p)).dot(Ntj));
        I1 = GCTriInt(p,Vj1,Vj2,RowVector3d(0,0,0));
        II1 = GCTriInt(RowVector3d(0,0,0),Vj2,Vj1,RowVector3d(0,0,0));
        q1 = Vj2.cross(Vj1);
        N1 = q1.normalized();
        
        // l=2
        double s2,I2,II2;
        RowVector3d q2,N2;
        s2 = copysign(1.0, ((Vj2-p).cross(Vj3-p)).dot(Ntj));
        I2 = GCTriInt(p,Vj2,Vj3,RowVector3d(0,0,0));
        II2 = GCTriInt(RowVector3d(0,0,0),Vj3,Vj2,RowVector3d(0,0,0));
        q2 = Vj3.cross(Vj2);
        N2 = q2.normalized();
        
        // l=3
        double s3,I3,II3;
        RowVector3d q3,N3;
        s3 = copysign(1.0, ((Vj3-p).cross(Vj1-p)).dot(Ntj));
        I3 = GCTriInt(p,Vj3,Vj1,RowVector3d(0,0,0));
        II3 = GCTriInt(RowVector3d(0,0,0),Vj1,Vj3,RowVector3d(0,0,0));
        q3 = Vj1.cross(Vj3);
        N3 = q3.normalized();
        
        double I;
        I = -abs(s1*I1+s2*I2+s3*I3);
        psi(j) = -I;
        RowVector3d w;
        w = Ntj*I + (N1*II1+N2*II2+N3*II3);
        
        if (w.norm()>0.0000000001) {
            phi(F_box(j,0)) = phi(F_box(j,0)) + (N2.dot(w))/(N2.dot(Vj1));
            phi(F_box(j,1)) = phi(F_box(j,1)) + (N3.dot(w))/(N3.dot(Vj2));
            phi(F_box(j,2)) = phi(F_box(j,2)) + (N1.dot(w))/(N1.dot(Vj3));
        }
        
    }
    return {phi,psi};
    
}






// **************************************************** //
// ********************** MAIN ************************ //
// **************************************************** //

int main(int argc, char *argv[]) {
    
    // Vertices
    MatrixXd V,VB,meshV1,cageV1(16,3),currMesh_V,currCage_V,newMesh_V,targetCage_V,cageV2,meshV2,cageV3,cageV4;
    
    // Faces
    MatrixXi F,FB,meshF1,cageF1(28,3),currMesh_F,currCage_F,newMesh_F,targetCage_F,cageF2,meshF2,cageF3,cageF4;
    
    // Edges
    MatrixXi E,cageE1(40,2),cageE2,cageE3,cageE4,currMesh_E,currCage_E,newMesh_E,targetCage_E;
    
    // Coefficient initialization
    int modelIdx = 0;
    
    // Load a mesh in OFF format
    igl::readOFF(TUTORIAL_SHARED_PATH "/bunny.off", meshV1, meshF1);
    igl::readOFF(TUTORIAL_SHARED_PATH "/spot_triangulated.off", meshV2, meshF2);
    igl::readOFF(TUTORIAL_SHARED_PATH "/spot_control_mesh_tri_scaled_test.off", cageV2, cageF2);
    igl::readOFF(TUTORIAL_SHARED_PATH "/Mytest2.off", cageV3, cageF3);
    igl::readOFF(TUTORIAL_SHARED_PATH "/mytest.off", cageV4, cageF4);
    
    // Visualize the mesh in a viewer
    igl::viewer::Viewer viewer;
    {
        viewer.core.show_lines = false;
        viewer.core.show_overlay_depth = false;
    }

    cageE2 = GetEdges(cageF2,cageV2.rows());
    cageE3 = GetEdges(cageF3,cageV3.rows());
    cageE4 = GetEdges(cageF4,cageV4.rows());
    
    // Define Cage5 MANUALLY:
    cageV1 <<  -0.106331, 0.0230912, -0.067873,
    0.07272625,  0.0230912,  -0.067873,
    0.07272625,  0.0930912,  -0.067873,
    -0.106331, 0.1130912, -0.067873,
    -0.106331,  0.0230912,  0.0671816,
    0.07272625,  0.0230912,  0.0671816,
    0.07272625,  0.0930912,  0.0671816,
    -0.106331,  0.1130912,  0.0671816,
    0.03272625,  0.1330912,  -0.067873,
    0.03272625,  0.1330912,  0.0671816,
    0.02272625,   0.201154,  -0.067873,
    0.02272625,   0.201154,  0.0671816,
    -0.106331,   0.181154,  0.0671816,
    -0.106331,   0.181154,  -0.067873,
    0.05272625,  0.0230912,  0.0671816,
    0.05272625,  0.0230912,  -0.067873;
    
    cageE1 <<
    0, 1,
    1, 2,
    2, 8,
    3, 8,
    3, 0,
    4, 5,
    5, 6,
    6, 9,
    7, 9,
    7, 4,
    0, 4,
    1, 5,
    2, 6,
    7, 3,
    3, 4,
    0, 8,
    15, 2,
    2, 5,
    5, 9,
    14, 7,
    5, 0,
    8, 9,
    8, 6,
    11, 9,
    10, 8,
    11, 10,
    12, 11,
    13, 10,
    3, 13,
    7, 12,
    12, 13,
    10, 9,
    3, 10,
    7, 13,
    9, 12,
    11, 13,
    9, 14,
    8, 15,
    0, 14,
    5, 15;
    
    cageF1 <<
    0, 4, 3,
    4, 7, 3,
    15, 2, 1,
    15, 8, 2,
    0, 8, 15,
    0, 3, 8,
    1, 2, 5,
    5, 2, 6,
    5, 6, 9,
    5, 9, 14,
    14, 9, 7,
    14, 7, 4,
    6, 8, 9,
    6, 2, 8,
    5, 14, 0,
    14, 4, 0,
    5, 0, 15,
    5, 15, 1,
    8, 10, 9,
    9, 10, 11,
    3, 10, 8,
    3, 13, 10,
    3, 7, 13,
    7, 12, 13,
    9, 11, 12,
    9, 12, 7,
    11, 10, 13,
    11, 13, 12;
    
    
    //  Deform the cage by translating a selected vertex
    double tx,ty,tz;
    int vertexInd;
    vertexInd = 0;
    tx = 0;
    ty = 0;
    tz = 0;
    
    V = meshV1;
    F = meshF1;
    
    // Show initial mesh
    viewer.data.clear();
    viewer.data.set_mesh(V,F);
    
    // Viewer menu
    viewer.callback_init =
    [&V,&meshV1,&meshV2,&cageV1,&cageV2,&cageV3,&cageV4,&currMesh_V,&currCage_V,&newMesh_V,&targetCage_V,&F,&meshF1,&meshF2,&cageF1,&cageF2,&cageF3,&cageF4,&currMesh_F,&currCage_F,&newMesh_F,&targetCage_F,&E,&cageE1,&currMesh_E,&currCage_E,&newMesh_E,&targetCage_E,&tx,&ty,&tz,&vertexInd,&cageE2,&cageE3,&cageE4,&modelIdx](igl::viewer::Viewer& viewer)
    {
        // Add an additional menu window
        viewer.ngui->addWindow(Vector2i(900,10), "Project: Cage-based Deformations");
        
        
        
        // ********************************************************************* //
        // ************************** Step1: Select Models ********************* //
        // ********************************************************************* //
        viewer.ngui->addGroup("Step1: Select Models");
        
        //***********************************************************************
        // Load Bunny Mesh
        //***********************************************************************
        viewer.ngui->addButton("Bunny",[&]() {
                                   V = meshV1;
                                   F = meshF1;
                                   newMesh_V = V;
                                   newMesh_F = F;
            
                                   modelIdx = 0;
                                   
                                   viewer.data.clear();
                                   viewer.data.set_mesh(V,F);
                               }
                               );
        
        //************************************************************************
        // Load Cow Mesh
        //************************************************************************
        viewer.ngui->addButton("Cow",[&]() {
                                    V = meshV2;
                                    F = meshF2;
                                    newMesh_V = V;
                                    newMesh_F = F;
            
                                    modelIdx = 1;
        
                                    viewer.data.clear();
                                    viewer.data.set_mesh(V,F);
                                }
                                );
        
        
        
        // ********************************************************************* //
        // ******************** Step2: Show Original Cages ********************* //
        // ********************************************************************* //
        viewer.ngui->addGroup("Step2: Show Original Cages");
        
        //**********************************************************************
        // Drawing Cages
        //**********************************************************************
        viewer.ngui->addButton("Corresponded Original Cages",[&]() {
                                    if (modelIdx == 0){
                                        currCage_V = cageV1;
                                        currCage_F = cageF1;
                                        currCage_E = cageE1;
                                    }else{
                                        currCage_V = cageV2;
                                        currCage_F = cageF2;
                                        currCage_E = cageE2;
                                    }
                                   
                                   // Plot the edges of the bounding box
                                   for (unsigned i=0;i<currCage_E.rows(); ++i)
                                       viewer.data.add_edges
                                       (
                                        currCage_V.row(currCage_E(i,0)),
                                        currCage_V.row(currCage_E(i,1)),
                                        RowVector3d(1,0,0)
                                        );
                                    }
                                    );

        
        
 
        // ********************************************************************* //
        // ******************** Step3: Select Target Cage ********************** //
        // ********************************************************************* //
        viewer.ngui->addGroup("Step3: Select Target Cage");
        
        //**********************************************************************
        // Define Deformed Cages for Bunny
        //**********************************************************************
        viewer.ngui->addGroup("For Bunny (single point control)");
        viewer.ngui->addVariable<int>("Vertex Index(0~15)", [&] (int val) {
                                          vertexInd = val;
                                      }, [&]() {
                                          return vertexInd;
                                      });
        //***********************************************************************
        viewer.ngui->addVariable<double>("Tranlsation along X-axis",[&] (double val) {
                                             tx = val;
                                         }, [&]() {
                                             return tx;
                                         });
        //************************************************************************
        viewer.ngui->addVariable<double>("Tranlsation along Y-axis", [&] (double val) {
                                             ty = val;
                                         }, [&]() {
                                             return ty;
                                         });
        //*************************************************************************
        viewer.ngui->addVariable<double>("Tranlsation along Z-axis",[&] (double val) {
                                             tz = val;
                                         }, [&]() {
                                             return tz;
                                         });
        //**************************************************************************
        viewer.ngui->addButton("Deformed Cage for Bunny",[&]() {
                                   RowVector3d translation;
                                   translation << tx,ty,tz;
                                   
                                   targetCage_V = cageV1;
                                   targetCage_V.row(vertexInd) = targetCage_V.row(vertexInd)+translation;
                                   targetCage_F = cageF1;
                                   targetCage_E = cageE1;
                                   
                                   viewer.data.clear();
                                   viewer.data.set_mesh(meshV1,meshF1);
            
                                   // Plot the corners of the cage as points
                                   MatrixXd vertColors;
                                   vertColors = RowVector3d(1,0,0).replicate(targetCage_V.rows(), 1);
                                   vertColors.row(vertexInd) = RowVector3d(0,0,1);
                                   
                                   // Plot the edges of the cage
                                   for (unsigned i=0;i<targetCage_E.rows(); ++i)
                                       viewer.data.add_edges
                                       (
                                        targetCage_V.row(targetCage_E(i,0)),
                                        targetCage_V.row(targetCage_E(i,1)),
                                        RowVector3d(0,1,0)
                                        );
                               });
        
        
        
        //**********************************************************************
        // Show Defined Deformation Cages for Cow
        //**********************************************************************
        viewer.ngui->addGroup("For Cow (partial control)");
        viewer.ngui->addButton("Deformed Cage1 for Cow",[&]() {
                                   targetCage_V = cageV3;
                                   targetCage_F = cageF3;
                                   targetCage_E = cageE3;
                                   
                                   viewer.data.clear();
                                   viewer.data.set_mesh(meshV2,meshF2);
            
                                   // Plot the corners of the bounding box as points
                                   MatrixXd vertColors;
                                   vertColors = RowVector3d(1,0,0).replicate(targetCage_V.rows(), 1);
                                   
                                   // Plot the edges of the bounding box
                                   for (unsigned i=0;i<targetCage_E.rows(); ++i)
                                       viewer.data.add_edges
                                       (
                                        targetCage_V.row(targetCage_E(i,0)),
                                        targetCage_V.row(targetCage_E(i,1)),
                                        RowVector3d(0,1,0)
                                        );
                               });
        //**************************************************************************
        viewer.ngui->addButton("Deformed Cage2 for Cow",[&]() {
            targetCage_V = cageV4;
            targetCage_F = cageF4;
            targetCage_E = cageE4;
            
            viewer.data.clear();
            viewer.data.set_mesh(meshV2,meshF2);
            
            // Plot the corners of the cage as points
            MatrixXd vertColors;
            vertColors = RowVector3d(1,0,0).replicate(targetCage_V.rows(), 1);
            
            // Plot the edges of the cage
            for (unsigned i=0;i<targetCage_E.rows(); ++i)
                viewer.data.add_edges
                (
                 targetCage_V.row(targetCage_E(i,0)),
                 targetCage_V.row(targetCage_E(i,1)),
                 RowVector3d(0,1,0)
                 );
        });
        
        

        
        
        // ********************************************************************* //
        // ****************** Step4: Select Deformation Mode ******************* //
        // ********************************************************************* //

        viewer.ngui->addGroup("Step4: Select Deformation Mode");
        //***********************************************************************
        // Mean-Coords Deformation
        //***********************************************************************
        viewer.ngui->addButton("Mean-Coords Deforming",[&]() {
                                   // Start time counting
                                   auto start = system_clock::now();
            
                                   // Iterate each vertex and compute weights by mean-coords
                                   for (int i=0; i<V.rows(); i++) {
                                       RowVector3d q;
                                       q = V.row(i);
                                       VectorXd Weights;
                                       Weights = MeanCoords(currCage_V,currCage_F,q);
                                       newMesh_V.row(i) = Weights.transpose()*targetCage_V;
                                   }
            
                                   // End time counting, record processing time
                                   auto end = system_clock::now();
                                   auto duration = duration_cast<microseconds>(end - start);
                                   cout<< "Processing Time: " <<double(duration.count()) * microseconds::period::num / microseconds::period::den<< "s" << endl;
                                   
                                   viewer.data.clear();
                                   viewer.data.set_mesh(newMesh_V,newMesh_F);
                                   
                                   // Plot the edges of the cage
                                   for (unsigned i=0;i<targetCage_E.rows(); ++i)
                                       viewer.data.add_edges
                                       (
                                        targetCage_V.row(targetCage_E(i,0)),
                                        targetCage_V.row(targetCage_E(i,1)),
                                        RowVector3d(0,1,0)
                                        );
                               });
        
        //***********************************************************************
        // Green-Coords Deformation
        //***********************************************************************
        viewer.ngui->addButton("Green-Coords Deforming",[&]() {
            
                                   MatrixXd normals,newNormals;
                                   igl::per_face_normals(currCage_V,currCage_F,normals);
                                   igl::per_face_normals(targetCage_V, targetCage_F,newNormals);
            
                                   // Start time counting
                                   auto start = system_clock::now();
            
                                   // Iterate each vertex and compute weights by green-coords
                                   for (int i=0; i<V.rows(); i++) {
                                       RowVector3d q;
                                       q = V.row(i);
                                       RowVectorXd phi,psi;
                                       auto params = GreenCoords(currCage_V, currCage_F, normals, q);
                                       phi = get<0>(params);
                                       psi = get<1>(params);
                                       newMesh_V.row(i) = phi*targetCage_V + psi*newNormals;
                                   }
            
                                   // End counting and record processing time
                                   auto end = system_clock::now();
                                   auto duration = duration_cast<microseconds>(end - start);
                                   cout<< "Processing Time: " <<double(duration.count()) * microseconds::period::num / microseconds::period::den<< "s" << endl;
                                   
                                   viewer.data.clear();
                                   viewer.data.set_mesh(newMesh_V,newMesh_F);
                                   
                                   // Plot the edges of the cage
                                   for (unsigned i=0;i<targetCage_E.rows(); ++i)
                                       viewer.data.add_edges
                                       (
                                        targetCage_V.row(targetCage_E(i,0)),
                                        targetCage_V.row(targetCage_E(i,1)),
                                        RowVector3d(0,1,0)
                                        );
                               });
        
        
       
        //**************************************************************************
        viewer.screen->performLayout();
        return false;
    };
    
    viewer.launch();
    return 0;
}
