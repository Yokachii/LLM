#include "../include/dataset.h"
#include "../include/perceptron.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>

int main(int argc, char *argv[]) {

    if (argc<3) {
        fprintf(stderr, "Usage: %s <modele_path> <text_path> <seed>\n", argv[0]);
        fprintf(stderr, "Example: %s ./modeles/main1.txt ./datasets/full.txt 725462\n", argv[0]);
        return 1;
    }

    char *model_path = argv[1];
    char *text_path = argv[2];
    unsigned int seed = (argc > 3) ? (unsigned int)strtoul(argv[3], NULL, 10) : (unsigned int)time(NULL);
    
    srand(seed);

    
    Modele *m = charger_modele(model_path);
    if (m == NULL) {
        fprintf(stderr, "Erreur : impossible de charger le modèle '%s'\n", model_path);
        return 1;
    }
    Vocab *voc = m->voc;
    Reseau *net = m->net;

    Texte *text = encoder_texte(text_path, voc);
    if (text == NULL) {
        free_modele(m);
        free_network(net);
        free_vocab(voc);
        return 1;
    }

    // Valeurs du modèle :
    displayPerfs(m,text);

    free_modele(m);
    free_network(net);
    free_text(text);
    free_vocab(voc);
    return 0;
}