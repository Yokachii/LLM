#include "../include/dataset.h"
#include "../include/perceptron.h"
#include <stdlib.h>
#include <math.h>

int main(int argc, char *argv[]) {

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <modele_path> <taille> <temp> <amorce>\n", argv[0]);
        fprintf(stderr, "Example: %s ./modeles/M1.txt 2000 0.5 \"Je suis un\"\n", argv[0]);
        return 1;
    }

    char *filepath = argv[1];

    int taille = (argc > 2) ? atoi(argv[2]) : 2000;
    double temp = (argc > 3) ? atof(argv[3]) : 0.5;

    Modele *m = charger_modele(filepath);
    if (m == NULL) {
        fprintf(stderr, "Erreur : impossible de charger le modèle '%s'\n", filepath);
        return 1;
    }

    Reseau *net = m->net;
    Vocab *voc = m->voc;
    int *amorce_tok = malloc((size_t)m->N * sizeof(*amorce_tok));
    if (amorce_tok == NULL) {
        fprintf(stderr, "Erreur : allocation impossible pour l'amorce\n");
        free_modele(m);
        free_network(net);
        free_vocab(voc);
        return 1;
    }
    encoder_amorce(m->voc, argv[4], m->N, amorce_tok);

    generer(m,taille,temp,amorce_tok);
    printf("\n\n");

    free_modele(m);
    free_network(net);
    free_vocab(voc);
    free(amorce_tok);
    
    return 0;
}