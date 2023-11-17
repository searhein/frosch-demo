// Copyright 2021 Alexander Heinlein
// Contact: Alexander Heinlein (a.heinlein@tudelft.nl)

#ifndef _HEADERS_AND_HELPERS_
#define _HEADERS_AND_HELPERS_

// std
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <exception>

#ifdef HAVE_Boost
#include <boost/filesystem.hpp>
#endif

#ifdef HAVE_VTK
#include <vtkVersion.h>
#include <vtkSmartPointer.h>
#include <vtkCellArray.h>
#include <vtkRectilinearGrid.h>
#include <vtkXMLPRectilinearGridWriter.h>
#include <vtkXMLRectilinearGridWriter.h>
#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkDataArray.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkMath.h>
#include <vtkDoubleArray.h>
#endif

// Belos
#include <BelosLinearProblem.hpp>
#include "BelosBlockGmresSolMgr.hpp"
#include "BelosPseudoBlockGmresSolMgr.hpp"
#include "BelosBlockCGSolMgr.hpp"
#include "BelosPseudoBlockCGSolMgr.hpp"
#include <BelosSolverFactory.hpp>
#include <BelosTpetraAdapter.hpp>

// Galeri::Xpetra
#include "Galeri_XpetraProblemFactory.hpp"
#include "Galeri_XpetraMatrixTypes.hpp"
#include "Galeri_XpetraParameters.hpp"
#include "Galeri_XpetraUtils.hpp"
#include "Galeri_XpetraMaps.hpp"

// Teuchos
#include <Teuchos_Array.hpp>
#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_StackedTimer.hpp>
#include <Teuchos_Tuple.hpp>

// Thyra
#include <Thyra_LinearOpWithSolveBase.hpp>
#include <Thyra_VectorBase.hpp>
#include <Thyra_SolveSupportTypes.hpp>
#include <Thyra_LinearOpWithSolveBase.hpp>
#include <Thyra_LinearOpWithSolveFactoryHelpers.hpp>
#include <Thyra_TpetraLinearOp.hpp>
#include <Thyra_TpetraMultiVector.hpp>
#include <Thyra_TpetraVector.hpp>
#include <Thyra_TpetraThyraWrappers.hpp>
#include <Thyra_VectorBase.hpp>
#include <Thyra_VectorStdOps.hpp>
#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
#include <Thyra_EpetraLinearOp.hpp>
#endif
#include <Thyra_VectorSpaceBase_def.hpp>
#include <Thyra_VectorSpaceBase_decl.hpp>

// Xpetra
#include <Xpetra_Map.hpp>
#include <Xpetra_CrsMatrixWrap.hpp>
#include <Xpetra_DefaultPlatform.hpp>
#ifdef HAVE_SHYLU_DDFROSCH_EPETRA
#include <Xpetra_EpetraCrsMatrix.hpp>
#endif
#include <Xpetra_Parameters.hpp>

// FROSch
#include <ShyLU_DDFROSch_config.h>
#include <FROSch_Tools_def.hpp>
#include <FROSch_SchwarzPreconditioners_fwd.hpp>
#include <FROSch_OneLevelPreconditioner_def.hpp>

// namespaces
using namespace Belos;
using namespace FROSch;
using namespace std;
using namespace Teuchos;
using namespace Xpetra;

// typedefs
typedef MultiVector<double,int,FROSch::DefaultGlobalOrdinal,KokkosClassic::DefaultNode::DefaultNodeType> multivector_type;
typedef multivector_type::scalar_type scalar_type;
typedef multivector_type::local_ordinal_type local_ordinal_type;
typedef multivector_type::global_ordinal_type global_ordinal_type;
typedef multivector_type::node_type node_type;
typedef MultiVectorFactory<scalar_type,local_ordinal_type,global_ordinal_type,node_type> multivectorfactory_type;
typedef Map<local_ordinal_type,global_ordinal_type,node_type> map_type;
typedef Matrix<scalar_type,local_ordinal_type,global_ordinal_type,node_type> matrix_type;
typedef CrsMatrixWrap<scalar_type,local_ordinal_type,global_ordinal_type,node_type> crsmatrixwrap_type;

typedef Galeri::Xpetra::Problem<Map<local_ordinal_type,global_ordinal_type,node_type>,crsmatrixwrap_type,multivector_type> problem_type;

typedef Belos::OperatorT<multivector_type> operatort_type;
typedef Belos::LinearProblem<scalar_type,multivector_type,operatort_type> linear_problem_type;
typedef Belos::SolverFactory<scalar_type,multivector_type,operatort_type> solverfactory_type;
typedef Belos::SolverManager<scalar_type,multivector_type,operatort_type> solver_type;
typedef XpetraOp<scalar_type,local_ordinal_type,global_ordinal_type,node_type> xpetraop_type;

typedef FROSch::OneLevelPreconditioner<scalar_type,local_ordinal_type,global_ordinal_type,node_type> onelevelpreconditioner_type;
typedef FROSch::TwoLevelPreconditioner<scalar_type,local_ordinal_type,global_ordinal_type,node_type> twolevelpreconditioner_type;

int assembleSystemMatrix (RCP<const Comm<int> > comm,
                          UnderlyingLib xpetraLib,
                          string equation,
                          int dimension,
                          int N,
                          int M,
                          RCP<matrix_type> &A,
                          RCP<multivector_type> &coordinates)
{
    // Create parameter list for Galeri
    ParameterList GaleriList;
    global_ordinal_type n = N;
    global_ordinal_type m = N*M;
    GaleriList.set("nx",m); GaleriList.set("ny",m); GaleriList.set("nz",m);
    GaleriList.set("mx",n); GaleriList.set("my",n); GaleriList.set("mz",n);

    RCP<const map_type> nodeMap, dofMap;
    RCP<problem_type> problem;
    if (dimension == 2) {
        nodeMap = Galeri::Xpetra::CreateMap<local_ordinal_type,global_ordinal_type,node_type>(xpetraLib,"Cartesian2D",comm,GaleriList);
        coordinates = Galeri::Xpetra::Utils::CreateCartesianCoordinates<scalar_type,local_ordinal_type,global_ordinal_type,map_type,multivector_type>("2D",nodeMap,GaleriList);
        if (!equation.compare("laplace")) {
            dofMap = nodeMap;
            problem = Galeri::Xpetra::BuildProblem<scalar_type,local_ordinal_type,global_ordinal_type,map_type,crsmatrixwrap_type,multivector_type>("Laplace2D",dofMap,GaleriList);
        } else {
            dofMap = Xpetra::MapFactory<local_ordinal_type,global_ordinal_type,node_type>::Build(nodeMap,2);
            problem = Galeri::Xpetra::BuildProblem<scalar_type,local_ordinal_type,global_ordinal_type,map_type,crsmatrixwrap_type,multivector_type>("Elasticity2D",dofMap,GaleriList);
        }
        A = problem->BuildMatrix();
    } else if (dimension == 3) {
        nodeMap = Galeri::Xpetra::CreateMap<local_ordinal_type,global_ordinal_type,node_type>(xpetraLib,"Cartesian3D",comm,GaleriList);
        coordinates = Galeri::Xpetra::Utils::CreateCartesianCoordinates<scalar_type,local_ordinal_type,global_ordinal_type,map_type,multivector_type>("3D",nodeMap,GaleriList);
        if (!equation.compare("laplace")) {
            dofMap = nodeMap;
            problem = Galeri::Xpetra::BuildProblem<scalar_type,local_ordinal_type,global_ordinal_type,map_type,crsmatrixwrap_type,multivector_type>("Laplace3D",dofMap,GaleriList);
        } else {
            dofMap = Xpetra::MapFactory<local_ordinal_type,global_ordinal_type,node_type>::Build(nodeMap,3);
            problem = Galeri::Xpetra::BuildProblem<scalar_type,local_ordinal_type,global_ordinal_type,map_type,crsmatrixwrap_type,multivector_type>("Elasticity3D",dofMap,GaleriList);
        }
        A = problem->BuildMatrix();
    }
    return 0;
}

#if defined (HAVE_VTK) && defined (HAVE_Boost)
template<int dimension, int dofsPerNode>
int writeSolVTK (string type,
                 int N,
                 int M,
                 const Teuchos::ArrayView<const double> &p0,
                 RCP<multivector_type> x)
{
    FROSCH_ASSERT(!x.is_null(),"x is null.");

    string output_foldername = string("./out-") + type;
    char number[9];
    sprintf (number, "%07d", x->getMap()->getComm()->getRank());
    #ifdef HAVE_Boost
        const boost::filesystem::path dir(output_foldername);
        if (dir.string() != ".")
        boost::filesystem::create_directories(dir);
    #endif

    string filename_sol = string("./") + output_foldername + string("/sol_") + number + string(".vtr");

    vtkSmartPointer<vtkDoubleArray> xpoints = vtkSmartPointer<vtkDoubleArray>::New();
    vtkSmartPointer<vtkDoubleArray> ypoints = vtkSmartPointer<vtkDoubleArray>::New();
    vtkSmartPointer<vtkDoubleArray> zpoints = vtkSmartPointer<vtkDoubleArray>::New();
    for (size_t i = 0; i < M+1; i++) {
        xpoints->InsertNextValue(p0[0]+double(i)/(M*N-1));
        ypoints->InsertNextValue(p0[1]+double(i)/(M*N-1));
    }

    vtkNew<vtkRectilinearGrid> grid;
    if (dimension == 2) {
        grid->SetDimensions(M+1, M+1, 1);

        zpoints->InsertNextValue(0.0);
    } else if (dimension == 3) {
        grid->SetDimensions(M+1, M+1, M+1);

        for (size_t i = 0; i < M+1; i++) {
            zpoints->InsertNextValue(p0[2]+double(i)/(M*N-1));
        }
    }
    grid->SetXCoordinates(xpoints);
    grid->SetYCoordinates(ypoints);
    grid->SetZCoordinates(zpoints);

    vtkSmartPointer<vtkDoubleArray> solution_array = vtkSmartPointer<vtkDoubleArray>::New();
    solution_array->SetNumberOfComponents(dofsPerNode);
    solution_array->SetName("u");
    for (int i = 0; i < x->getLocalLength()/dofsPerNode; i++) {
        solution_array->InsertNextTuple(&x->getData(0)[dofsPerNode*i]);
    }
    grid->GetCellData()->AddArray(solution_array);

    vtkNew<vtkXMLRectilinearGridWriter> writer;
    writer->SetFileName(filename_sol.c_str());
    writer->SetInputData(grid);
    writer->Write();

    return 0;
}

int writeVTK (string equation,
              int dimension,
              int N,
              int M,
              RCP<multivector_type> coordinates,
              RCP<multivector_type> x)
{
    using Teuchos::tuple;

    FROSCH_ASSERT(!coordinates.is_null(),"coordinates is null.");
    FROSCH_ASSERT(!x.is_null(),"x is null.");

    if (dimension == 2) {
        const double x0 = coordinates->getData(0)[0], y0 = coordinates->getData(1)[0];
        if (!equation.compare("laplace")) {
            return writeSolVTK<2,1>("laplace2d",N,M,tuple<double>(x0,y0),x);
        } else {
            return writeSolVTK<2,2>("elasticity2d",N,M,tuple<double>(x0,y0),x);
        }
    } else if (dimension == 3) {
        const double x0 = coordinates->getData(0)[0], y0 = coordinates->getData(1)[0], z0 = coordinates->getData(2)[0];
        if (!equation.compare("laplace")) {
            return writeSolVTK<3,1>("laplace3d",N,M,tuple<double>(x0,y0,z0),x);
        } else {
            return writeSolVTK<3,3>("elasticity3d",N,M,tuple<double>(x0,y0,z0),x);
        }
    }
    return -1;
}
#endif

#endif
