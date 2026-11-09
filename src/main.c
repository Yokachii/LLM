#include "../include/dataset.h"
#include "../include/perceptron.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>

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

    Vocab *voc;
    Texte *text_l = load_texte(filepath,&voc,d); // <- load text & set le nb de caractères distinct dans le text
    int dimE = N*d;
    Reseau *n = new_network(nb_couche_cache, H, taux, dimE, voc->V);
    Modele *m = new_modele(N,voc,n);
    Texte *tr, *tt;

    // display_Vocab(*voc);
    
    split_texte(text_l,&tr,&tt,train_ratio);

    int isFreed = 0;

    if(tt->n==0){
        free_text(tt);
        tt=tr;
        isFreed=1;
    }

    // Valeurs initiales du modèle :
    printf("perte initiale val = %.4f\n", calc_perte(m, tt, 1));
    printf("ln(V) = %.4f\n", log((double)voc->V));

    // Création initiale : 
    // generer(m,25,2,get_indices(m,"Bonjour toi",N));
    int amorce[8] = {1,4,2,3,1,5,2,3};
    printf("Texte généré avant entrainement : ");
    generer(m,25,0.5,amorce);
    printf("\n\n");



    // Train
    // train(m, nb_epoques, *tr, HumanOutput);
    train_modele(m,tr,tt,nb_epoques,taux);
    // display_Text(*tr);

    printf("Texte généré après entrainement : ");
    generer(m,25,0.5,amorce);
    printf("\n\n");


    if(isFreed==0){
        free_text(tt);
    }

    free_network(n);
    free_text(tr);
    free_vocab(voc);
    return 0;
}