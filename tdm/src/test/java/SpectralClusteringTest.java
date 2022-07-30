import com.mass.clustering.SpectralClustering;
import org.apache.commons.lang3.tuple.Pair;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;

public class SpectralClusteringTest {

    @Test
    public void testCluster() {
        int numClusters = 2;
        int row = 100;
        int col = 64;
        Random random = new Random();
        double[][] data = new double[row][col];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                data[i][j] = random.nextDouble();
            }
        }
        Pair<double[], double[][]> result = SpectralClustering.fit(data, numClusters, 1.0, 10);
        assertEquals(numClusters, result.getKey().length);
        assertEquals(row, result.getValue().length);
        for (double[] embed : result.getValue()) {
            assertEquals(numClusters, embed.length);
        }
    }
}
