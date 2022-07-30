package com.mass.clustering;

import org.apache.commons.lang3.tuple.Pair;
import smile.clustering.KMeans;
import smile.math.MathEx;
import smile.math.blas.UPLO;
import smile.math.matrix.ARPACK;
import smile.math.matrix.Matrix;

// https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html
// https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering
public class SpectralClustering {

    public static Pair<double[], double[][]> fit(double[][] data, int k, double sigma, int maxIter) {
        if (k < 2) {
            throw new IllegalArgumentException("Invalid number of clusters: " + k);
        }

        if (sigma <= 0.0) {
            throw new IllegalArgumentException("Invalid standard deviation of Gaussian kernel: " + sigma);
        }

        Matrix m = constructMatrix(data, sigma);
        return fitMatrix(m, k, maxIter);
    }

    private static Matrix constructMatrix(double[][] data, double sigma) {
        int n = data.length;
        double gamma = -0.5 / (sigma * sigma);

        Matrix W = new Matrix(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                double w = Math.exp(gamma * MathEx.squaredDistance(data[i], data[j]));
                W.set(i, j, w);
                W.set(j, i, w);
            }
        }
        return W;
    }

    private static Pair<double[], double[][]> fitMatrix(Matrix m, int k, int maxIter) {
        int n = m.nrows();
        double[] D = m.colSums();
        for (int i = 0; i < n; i++) {
            if (D[i] == 0.0) {
                throw new IllegalArgumentException("Isolated vertex: " + i);
            }

            D[i] = 1.0 / Math.sqrt(D[i]);
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                double w = D[i] * m.get(i, j) * D[j];
                m.set(i, j, w);
                m.set(j, i, w);
            }
        }

        m.uplo(UPLO.LOWER);
        Matrix.EVD eigen = ARPACK.syev(m, ARPACK.SymmOption.LA, k);
        double[][] projected = eigen.Vr.toArray();
        for (int i = 0; i < n; i++) {
            MathEx.unitize2(projected[i]);
        }

        KMeans kmeans = KMeans.fit(projected, k, maxIter, 1e-4);
        return Pair.of(kmeans.centroids[0], projected);
    }
}
