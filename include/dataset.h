#ifndef DATASET_H
#define DATASET_H

#define ACT_SOFTMAX 1
#define ACT_TANH 2
#define ACT_SIG 3

typedef struct couche_s
{
    float *b;
    int act;
    int dim;
    float taux;
    float **poids;
    int nb_neuronnes;
} Couche;

typedef struct reseau_s {
    int nb_layer;
    Couche **Layers;
} Reseau;

typedef struct {
    int V;                       // nombre de caractères distincts
    int d;                       // taille du vecteur de chaque caractère
    int *dic;                // caractère -> indice, -1 si absent
    unsigned char *dic_inv;  // indice -> caractère
    float **emb;                 // emb[indice][j] : V lignes de d floats (appris)
} Vocab; // <- table d'embeding + vocab

typedef struct {
    int n;                      // nombre de tokens
    int *tok;         // n indices <- contient des indices de caractères (0..V-1)
    Vocab *voc; // Lien vers le vocabulaire
} Texte;

typedef struct {
    int N;                  // taille du contexte
    Vocab *voc;             // Vocab (V, d, emb)
    Reseau *net;            // Reseau (Layers[0..L-2] : tanh ; Layers[L-1] : softmax)
    float *x;               // N*d : entrée du réseau
    float *dx;              // N*d : gradient sur l'entrée
    float **a;              // a[l] : sortie de la couche l
    float **delta;          // delta[l] : même taille que a[l]
} Modele; // On alloue une fois les tableaux de travail pour le modèle, pour éviter de réallouer à chaque update.

Texte *new_text(int n, Vocab *voc);

void free_text(Texte *t);
void free_vocab(Vocab *v);

Texte *load_texte(char *path, Vocab **v, int d);


void display_Text(Texte t);
void display_Vocab(Vocab v);


float sigmoide(float z);

void split_texte(Texte *t, Texte **train, Texte **test, float ratio);

#endif