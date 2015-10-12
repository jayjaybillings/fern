// ------------------------------------------------------------------------------------------------------------------------
//  Class to test JAMA package.  See http://math.nist.gov/javanumerics/jama/
//  Jama implements a basic linear algebra package for java.
// ------------------------------------------------------------------------------------------------------------------------

import Jama.*;                    // The JAMA classes are in the Jama subdirectory

class testJAMA {

    public static void main(String[] args) {
    
	double[][] array = {{1.,2.,3},{4.,5.,6.},{7.,8.,10.}};
	Matrix A = new Matrix(array);
	Matrix b = Matrix.random(3,1);
	// Solve the matrix equation A.x = b for the vector x
	Matrix x = A.solve(b);
	// The difference between A.x and b
	Matrix r = A.times(x).minus(b);
	// Create an object that is the LU decomposition of the matrix A
	LUDecomposition luA = A.lu();
	// Extract the lower and upper triangular matrices from the LU decomposition
        // and the pivot vector giving the row permutations in the LU decomposition
	Matrix lowerA = luA.getL();
	Matrix upperA = luA.getU();
        int [] pivA = luA.getPivot();
	Matrix LUproduct = lowerA.times(upperA);
	// Inverse of the matrix A
	Matrix invA = A.inverse();
	// Determinant of A
	double detA = A.det();
	// Create an eigenvalue decomposition of A
	EigenvalueDecomposition eigen = A.eig();
	Matrix diag = eigen.getD();
	Matrix eigenvector = eigen.getV();
	double eigenvalReal [] = eigen.getRealEigenvalues();
        double eigenvalImag [] = eigen.getImagEigenvalues();

	System.out.println("\nMatrix A:");
	A.print(10,3);
	System.out.println("Column vector b:");
	b.print(10,3);
	System.out.println("Column vector solution x for A.x = b");
	x.print(10,3);
	System.out.println("Difference between A.x and b");
	r.print(20,17);
	System.out.println("Upper triangular matrix in LU decomposition of A:");
	upperA.print(10,3);
	System.out.println("Lower triangular matrix in LU decomposition of A:");
	lowerA.print(10,3);
        System.out.println("Row permutation in LU decomposition implied by pivot vector:\n");
        for(int i=0; i<pivA.length; i++){
            System.out.println(pivA[i]+"-->"+i);
        }
	System.out.println("\nMatrix product LU (= A up to row permutations---see pivot vector):");
	LUproduct.print(10,3);
	System.out.println("Inverse of matrix A:");
	invA.print(10,3);
	System.out.println("Product of A and its inverse:");
	A.times(invA).print(10,3);
	System.out.println("Det A = "+detA);
	System.out.println("\nA diagonalized:");
	diag.print(10,3);
	System.out.println("Eigenvectors matrix  of A:");
	eigenvector.print(10,3);
	for(int i=0; i<eigenvalReal.length; i++){
	    System.out.println("Real(eigenvalue "+i+") = "+eigenvalReal[i]);
	}
        System.out.println();
        for(int i=0; i<eigenvalImag.length; i++){
            System.out.println("Imaginary(eigenvalue "+i+") = "+eigenvalImag[i]);
        }
        
        // Create a new matrix to test LU decomposition (test is Maple linear algebra help file example)
        double[][] array2 = {{0,1,1,-3},{-2,3,1,4},{0,0,0,1},{3,1,0,0}};
        Matrix B = new Matrix(array2);
        System.out.println("\nMatrix B:");
        B.print(10,3);
        // Create an object that is the LU decomposition of the matrix B
        LUDecomposition luB = B.lu();
	// Extract the lower and upper triangular matrices from the LU decomposition
        // and the permutation matrix for pivots
        Matrix lowerB = luB.getL();
        Matrix upperB = luB.getU();
        int [] pivB = luB.getPivot();
        Matrix LUproductB = lowerB.times(upperB);
        System.out.println("Upper triangular matrix in LU decomposition of B:");
        upperB.print(10,3);
        System.out.println("Lower triangular matrix in LU decomposition of B:");
        lowerB.print(10,3);
        System.out.println("Row permutation in LU decomposition implied by pivot vector:\n");
        for(int i=0; i<pivB.length; i++){
            System.out.println(pivB[i]+"-->"+i);
        }
        System.out.println("\nMatrix product LU (= B up to row permutations---see pivot vector):");
        LUproductB.print(10,3);
        
        double [][] array3 = {{1,2,3,1}, {1,4,3,0}, {3,8,9,2}};
        Matrix C = new Matrix(array3);
        System.out.println("\nRank of the matrix");
        C.print(10,3);
        System.out.println("is "+C.rank()+"  (agrees with Boas, p. 91)");
        
        // Example from Boas pp. 103 and 108 ff.
        double [][] array4 = {{2,3,-1}, {1,1,1}, {-1,1,2}};
        Matrix D = new Matrix(array4);
        System.out.println("Solve following example from Boas, p. 103 and 108 ff: find x in Dx = k with D=");
        D.print(10,3);
        double [][] array5 = {{-3}, {2}, {2}};
        System.out.println(" and the vector k=");
        Matrix k = new Matrix(array5);
        k.print(10,3);
        System.out.println("Solution: x =");
        Matrix xx = D.solve(k);  
        xx.print(10,3);   
    } 
}
