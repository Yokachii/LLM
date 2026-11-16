#include "../include/dataset.h"
#include "../include/perceptron.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>

int main(int argc, char *argv[]) {

    if (argc<2) {
        fprintf(stderr, "Usage: %s <modele_path> <amorce>\n", argv[0]);
        fprintf(stderr, "Example: %s ./modeles/M1.txt Je suis un\n", argv[0]);
        return 1;
    }

    char *filepath = argv[1];

    srand(time(NULL));   // avant new_network, qui choisie les poids initiaux

    Modele *m = charger_modele(filepath);

    char *amorce_tok = NULL;
    encoder_amorce(m->voc,"TEST",4,amorce_tok);

    printf("Texte généré après entrainement à 0.5 : ");
    generer(m,2500,0.5,amorce_tok);
    printf("\n\n");

    printf("Texte généré après entrainement à 0.8: ");
    generer(m,2500,0.8,amorce_tok);
    printf("\n\n");

    printf("Texte généré après entrainement à 0.3 : ");
    generer(m,2500,0.3,amorce_tok);
    printf("\n\n");

    printf("Texte généré après entrainement à 1 : ");
    generer(m,2500,0.3,amorce_tok);
    printf("\n\n");

    free_network(m->net);
    free_vocab(m->voc);
    free_modele(m);
    
    return 0;
}