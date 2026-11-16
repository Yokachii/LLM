#include "../include/dataset.h"
#include "../include/perceptron.h"
#include <stdlib.h>
#include <math.h>

int main(int argc, char *argv[]) {

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <modele_path> <amorce>\n", argv[0]);
        fprintf(stderr, "Example: %s ./modeles/M1.txt \"Je suis un\"\n", argv[0]);
        return 1;
    }

    char *filepath = argv[1];

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
    encoder_amorce(m->voc, argv[2], m->N, amorce_tok);

    printf("Texte généré après entrainement à 0.5 : ");
    generer(m,2500,0.5f,amorce_tok);
    printf("\n\n");

    printf("Texte généré après entrainement à 0.8: ");
    generer(m,2500,0.8f,amorce_tok);
    printf("\n\n");

    printf("Texte généré après entrainement à 0.3 : ");
    generer(m,2500,0.3f,amorce_tok);
    printf("\n\n");

    printf("Texte généré après entrainement à 1 : ");
    generer(m,2500,1.0f,amorce_tok);
    printf("\n\n");

    free_modele(m);
    free_network(net);
    free_vocab(voc);
    free(amorce_tok);
    
    return 0;
}