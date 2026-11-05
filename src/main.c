#include "../include/dataset.h"
#include "../include/perceptron.h"
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {

    if (argc<2) {
        fprintf(stderr, "Usage: %s <dataset_path> <nb_epoques> <learning_rate> <train_ratio> <N> <d> <H> <couches_cache> <HumanOutput> <graine>\n", argv[0]);
        fprintf(stderr, "Example: %s ./datasets/corpus.txt 10 0.1 0.8 1 42\n", argv[0]);
        return 1;
    }

    char *filepath = argv[1];

    int nb_epoques = (argc > 2) ? atoi(argv[2]) : 5;
    float taux = (argc > 3) ? atof(argv[3]) : 1;
    float train_ratio = (argc > 4) ? atof(argv[4]) : 0.8;
    int N = (argc > 5) ? atoi(argv[5]) : 8;
    int d = (argc > 6) ? atoi(argv[6]) : 10;
    int H = (argc > 7) ? atoi(argv[7]) : 100;
    int nb_couche_cache = (argc > 8) ? atoi(argv[8]) : 2;
    int HumanOutput = (argc > 9) ? atoi(argv[9]) : 1;
    unsigned int seed = (argc > 10) ? (unsigned int)strtoul(argv[10], NULL, 10) : (unsigned int)time(NULL);
    

    srand(seed);   // avant new_network, qui choisie les poids initiaux
    if(HumanOutput==1) printf("Graine : %u\n", seed);

    int V = NULL;

    Dataset *dataset_l = load_dataset(filepath,&V); // <- load text & set le nb de caractères distinct dans le text

    int dimE = N*d;

    Reseau *n = new_network(nb_couche_cache, H, taux, dimE, V);
    Dataset *tr, *tt;
    shuffle(dataset_l);
    // if(normalizeArg==1){
    //     normalize(d);
    // }
    split_dataset(dataset_l,&tr,&tt,train_ratio);

    int isFreed = 0;

    if(tt->n==0){
        free_dataset(tt);
        tt=tr;
        isFreed=1;
    }

    // float accBefore=calc_acc(n,tt);
    // if(HumanOutput==1) printf("\nAcc avant train : %f\n\n",accBefore);

    // int epoquesMax;
    // float maxAcc;
    // int epoques = train(n,nb_epoques,tr,tt,StopWhenTrained,&epoquesMax,&maxAcc,HumanOutput);

    // MatrixConfusion *matrix = matrice_confusion(n,tt);
    // float accAfter=calc_acc(n,tt);

    // if(HumanOutput==1){
    //     display_network(*n);
    //     printf("Acc après train : %f\n\n",accAfter);
    //     printf("epoquesMax : %d\nAccMax:%f\n",epoquesMax,maxAcc);
    //     display_matrix(matrix);
    // }else{
    //     // #END;epoques;accBefore;accAfter;FN;FP;TN;TP;epoquesMax;accMax
    //     printf("#END;%d;%f;%f;%d;%d;%d;%d;%d;%f\n",epoques,accBefore,accAfter,matrix->FN,matrix->FP,matrix->TN,matrix->TP,epoquesMax,maxAcc);
    // }

    // free(matrix);

    if(isFreed==0){
        free_dataset(tt);
    }

    free_network(n);
    free_dataset(tr);
    return 0;
}