package com.mass.retrieval.tdm;

import com.mass.tdm.model.TDM;

import java.util.Arrays;

public class JavaRecommend {

    public static void main(String[] args) {
        String modelPath = "path/to/model";
        String treePath = "path/to/tree_pb_file";
        TDM tdmModel = TDM.loadModel(modelPath, "DeepFM");
        TDM.loadTree(treePath);

        // user interacted sequence with 10 items, recommend 3 items with 20 candidates
        int[] sequence = new int[] {0, 0, 2126, 204, 3257, 3439, 996, 1681, 3438, 1882};
        System.out.println("Recommendation result: " + Arrays.toString(tdmModel.recommend(sequence, 3, 20)));

        for (int i = 0; i < 10; i++) {
            tdmModel.recommend(sequence, 3, 20);
        }
        long start = System.nanoTime();
        for (int i = 0; i < 100; i++) {
            tdmModel.recommend(sequence, 3, 20);
        }
        long end = System.nanoTime();
        System.out.printf("Average recommend time: %.4fms", (end - start) * 10 / 1e9d);
    }
}
