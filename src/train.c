#include "../include/dataset.h"
#include "../include/perceptron.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>

int main(int argc, char *argv[]) {

    if (argc<3) {
        fprintf(stderr, "Usage: %s <modele_path> <text_path> <nb_epoques> <learning_rate_base> <train_ratio> <HumanOutput> <graine>\n", argv[0]);
        fprintf(stderr, "Example: %s ./modeles/main1.txt ./datasets/full.txt 20 0.04 0.8 1 725462\n", argv[0]);
        return 1;
    }

    char *model_path = argv[1];
    char *text_path = argv[2];

    int nb_epoques = (argc > 3) ? atoi(argv[3]) : 10;
    float taux = (argc > 4) ? atof(argv[4]) : 0.03;
    float train_ratio = (argc > 5) ? atof(argv[5]) : 0.8;
    int HumanOutput = (argc > 6) ? atoi(argv[6]) : 1;
    unsigned int seed = (argc > 7) ? (unsigned int)strtoul(argv[7], NULL, 10) : (unsigned int)time(NULL);
    
    srand(seed);
    if(HumanOutput==1) printf("Graine : %u\n", seed);

    
    Modele *m = charger_modele(model_path);
    if (m == NULL) {
        fprintf(stderr, "Erreur : impossible de charger le modèle '%s'\n", model_path);
        return 1;
    }
    Vocab *voc = m->voc;
    Reseau *net = m->net;

    Texte *text_l = encoder_texte(text_path, voc);
    if (text_l == NULL) {
        free_modele(m);
        free_network(net);
        free_vocab(voc);
        return 1;
    }
    Texte *tr, *tt;
    split_texte(text_l, &tr, &tt, train_ratio);

    int isFreed = 0;

    if(tt->n==0){
        free_text(tt);
        tt=tr;
        isFreed=1;
    }

    // Valeurs initiales du modèle :
    printf("perte initiale val = %.4f\n", calc_perte(m, tt, 1));
    printf("ln(V) = %.4f\n", log((double)voc->V));

    // Train
    train_modele(m,tr,tt,nb_epoques,taux,model_path);
    sauver_modele(m, model_path);

    printf("Entrainement terminé et model sauvegardé à : %s",model_path);


    if(isFreed==0){
        free_text(tt);
    }

    free_modele(m);
    free_network(net);
    free_text(tr);
    free_vocab(voc);
    return 0;
}