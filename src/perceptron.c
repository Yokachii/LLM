#include "../include/perceptron.h"
#include <stdlib.h>
#include <math.h>

void softmax(float *z, int n){
    float m = z[0];
    for (int i = 1; i < n; i++) if (z[i] > m) m = z[i];     // plus grand logit

    float somme = 0.0f;
    for (int i = 0; i < n; i++) {
        z[i] = expf(z[i] - m);                               // stable : exposant <= 0
        somme += z[i];
    }
    for (int i = 0; i < n; i++) z[i] /= somme;               // la somme des sorties vaut 1
}

Modele *new_modele(int n, Vocab *voc, Reseau *net){

    if (net->Layers[0]->dim != n * voc->d || net->Layers[net->nb_layer - 1]->nb_neuronnes != voc->V) {
        fprintf(stderr, "Erreur : dimensions du réseau incompatibles -> Changez le réseau\n");
        return NULL;
    }

    Modele *m = malloc(sizeof(Modele));
    
    m->N=n;
    m->voc=voc;
    m->net=net;
    m->x=malloc(sizeof(float)*n*voc->d);
    m->dx=malloc(sizeof(float)*n*voc->d);
    m->a=malloc(sizeof(float*)*net->nb_layer);
    m->delta=malloc(sizeof(float*)*net->nb_layer);

    for(int i = 0; i<net->nb_layer;i++){
        Couche *l = net->Layers[i];
        m->a[i]=malloc(sizeof(float)*l->nb_neuronnes);
        m->delta[i]=malloc(sizeof(float)*l->nb_neuronnes);
    }

    return m;
}

void free_modele(Modele *m){
    if (m == NULL) return;
    for (int i = 0; i < m->net->nb_layer; i++) {
        free(m->a[i]);
        free(m->delta[i]);
    }
    free(m->a);
    free(m->delta);
    free(m->x);
    free(m->dx);
    free(m);          // ne libère ni voc ni net
}

Reseau *new_network(int nb_couche_cache, int nb_neuronnes_ch, float taux, int dimE, int dimO){
    Reseau *res = malloc(sizeof(Reseau));

    res->nb_layer=2+nb_couche_cache;
    
    Couche **layers = malloc(sizeof(Couche*)*(res->nb_layer));

    int dimL0 = nb_neuronnes_ch;

    Couche *entree = new_layer(dimE,taux,dimL0,ACT_TANH);

    layers[0] = entree;

    for(int i = 0; i<nb_couche_cache;i++){
        int dimL = nb_neuronnes_ch;
        if(i==0) dimL=dimL0;
        Couche *l = new_layer(dimL,taux,nb_neuronnes_ch,ACT_TANH);
        layers[i+1] = l;
    }

    Couche *out = new_layer(nb_neuronnes_ch,taux,dimO,ACT_SOFTMAX);

    layers[res->nb_layer-1] = out;

    res->Layers=layers;

    return res;
}

void display_network(Reseau n){
    printf("Taux : %.2f\n",n.Layers[0]->taux);
    printf("\nCouche d'entré :\n");
    display_perceptron(*(n.Layers[0]));
    printf("\n\n");
    for(int i = 0; i<n.nb_layer-2;i++){
        printf("Couche N°%d (de %d neuronnes):\n",i+1,n.Layers[0]->nb_neuronnes);
        display_perceptron(*(n.Layers[i+1]));
        printf("\n\n");
    }
    printf("Couche de sortie :\n");
    display_perceptron(*(n.Layers[n.nb_layer-1]));
    
}

void free_network(Reseau *n){
    for(int i = 0;i<n->nb_layer;i++){
        free_layer(n->Layers[i]);
    }
    free(n);
}

Couche *new_layer(int dim, float taux, int nb_neuronnes, int act){
    Couche *n_perceptron = malloc(sizeof(Couche));

    n_perceptron->b = 0;
    n_perceptron->dim = dim;
    n_perceptron->taux = taux;
    n_perceptron->nb_neuronnes = nb_neuronnes;

    float *biais = malloc(sizeof(float)*nb_neuronnes);

    for(int i = 0; i<nb_neuronnes; i++){
        biais[i]=0;
    }

    n_perceptron->b=biais;


    float **poids = malloc(sizeof(float*)*nb_neuronnes);

    for(int i = 0; i<nb_neuronnes; i++){
        float *p = malloc(sizeof(float)*dim);

        float scale = sqrtf(1.0f / dim);
        for(int j = 0; j<dim; j++){
            p[j] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
        }

        poids[i]=p;
    }

    n_perceptron->poids=poids;

    n_perceptron->act=act;

    return n_perceptron;
};


void free_layer(Couche *p){
    int nb_neuronnes = p->nb_neuronnes;
    for(int i = 0;i<nb_neuronnes;i++){
        free(p->poids[i]);
    }
    free(p->b);
    free(p);
};

void display_perceptron(Couche p){
    printf("Dim : %d\n",p.dim);
    for(int j = 0; j<p.nb_neuronnes;j++){
        float b = p.b[j];
        int dim = p.dim;
        printf("Perceptron N°%d\nBiais : %.2f\n[\n    ",j+1,b);
        for(int i = 0; i<dim; i++){
            printf("%.2f, ",p.poids[j][i]);
        }
        printf("\n]\n\n");
    }
};

float* prediction(Modele *m, int indice_couche, float* inputs) {
    Couche *c = m->net->Layers[indice_couche];

    for (int j = 0; j < c->nb_neuronnes; j++) {
        float z = c->b[j]; // Biais
        for (int i = 0; i < c->dim; i++) z += c->poids[j][i] * inputs[i];
        m->a[indice_couche][j] = z;
    }
    if(c->act == ACT_SOFTMAX){
        softmax(m->a[indice_couche], c->nb_neuronnes);
    }else{
        for(int j = 0; j < c->nb_neuronnes; j++) m->a[indice_couche][j] = tanhf(m->a[indice_couche][j]);
    }
}

float *forward_modele(Modele *m, int *ctx){
    int d = m->voc->d;

    for (int k = 0; k < m->N; k++)
        for (int j = 0; j < d; j++)
            m->x[k*d + j] = m->voc->emb[ctx[k]][j];

    Reseau *net = m->net;
    float *inputs = m->x;
    for(int i = 0; i<net->nb_layer;i++){
        prediction(m,i,inputs);
    }

    return m->a[net->nb_layer - 1];
}

int argmax_proba(const float *p, int V){
    int best = 0;
    for (int k = 1; k < V; k++) if (p[k] > p[best]) best = k;
    return best;
}

// int calc_top_1(float *p, Modele *m, Texte *t){
//     return t->tok[argmax_proba(p, m->voc->V)];
// }

float calc_acc_top1(Modele *m, const Texte *t, int pas){
    int ok = 0, c = 0;
    for (int pos = m->N; pos < t->n; pos += pas) {
        float *p = forward_modele(m, t->tok + pos - m->N);
        if (argmax_proba(p, m->voc->V) == t->tok[pos]) ok++;
        c++;
    }
    return (float)ok / c;
}

float calc_perte(Modele *m, const Texte *t, int pas){   // pas = 1 : tous les exemples
    double L = 0;
    int c = 0;
    for (int pos = m->N; pos < t->n; pos += pas) {
        int *ctx = t->tok + pos - m->N;
        float *p = forward_modele(m, ctx);   // contexte = N indices avant pos
        float q = p[t->tok[pos]];            // probabilité du vrai caractère



        // printf("ctx : ");
        // for(int i = 0; i<m->N; i++) printf("%c",m->voc->dic_inv[ctx[i]]);
        // printf("\nProba pour chaque lettres :");
        // for(int i = 0; i<m->voc->V; i++){
        //     printf("\nProba pour '%c' : %f",m->voc->dic_inv[i],p[i]);
        // }
        // printf("\nMaxProba : '%c'\n\n",m->voc->dic_inv[argmax_proba(p,m->voc->V)]);



        if (q < 1e-12f) q = 1e-12f;                          // évite ln(0)
        L -= log(q);
        c++;
    }
    return (float)(L / c);
}

static void set_taux(Reseau *n, float taux){
    for(int l = 0; l < n->nb_layer; l++) n->Layers[l]->taux = taux;
}

void generer(Modele *m, int longueur, float temperature, const int *amorce){
    int N = m->N, V = m->voc->V;
    int *ctx = malloc(N * sizeof(int));
    double *q = malloc(V * sizeof(double));

    // Création du ctx initial
    for (int i = 0; i < N; i++) { ctx[i] = amorce[i]; putchar(m->voc->dic_inv[ctx[i]]); }

    for (int s = 0; s < longueur; s++) {
        float *p = forward_modele(m, ctx);
        double somme = 0;
        for (int k = 0; k < V; k++) { q[k] = pow(p[k], 1.0 / temperature); somme += q[k]; }
        double r = ((double)rand() / RAND_MAX) * somme, cumul = 0;
        int c = V - 1;
        for (int k = 0; k < V; k++) { cumul += q[k]; if (r <= cumul) { c = k; break; } }
        putchar(m->voc->dic_inv[c]);                    // indice -> caractère
        for (int i = 0; i < N - 1; i++) ctx[i] = ctx[i + 1];   // la fenêtre glisse
        ctx[N - 1] = c;
    }


    putchar('\n');
    free(ctx); free(q);
}

void   update_modele(Modele *m, const unsigned char *ctx, int cible) {
    Reseau *net = m->net;
    int L = net->nb_layer, N = m->N, d = m->voc->d;
    float *p = forward_modele(m, ctx);


    // delta de sortie
    int V = net->Layers[L-1]->nb_neuronnes;
    for (int k = 0; k < V; k++) m->delta[L-1][k] = p[k] - (k == cible ? 1.0f : 0.0f);
    
    // deltas des couches cachées (les poids n'ont pas encore changé)
    for (int l = L-2; l >= 0; l--) {
        Couche *cur = net->Layers[l], *next = net->Layers[l+1];
        for (int j = 0; j < cur->nb_neuronnes; j++) {
            float s = 0;
            for (int k = 0; k < next->nb_neuronnes; k++)
                s += next->poids[k][j] * m->delta[l+1][k];
            float a = m->a[l][j];
            m->delta[l][j] = s * (1.0f - a * a);        // dérivée de tanh
        }
    }
    
    
    // gradient par rapport à l'entrée : AVANT de modifier les poids de la couche 0
    Couche *c0 = net->Layers[0];
    for (int i = 0; i < c0->dim; i++) {
        float s = 0;
        for (int j = 0; j < c0->nb_neuronnes; j++) s += c0->poids[j][i] * m->delta[0][j];
        m->dx[i] = s;
    }

    // poids et biais de toutes les couches
    for (int l = 0; l < L; l++) {
        Couche *c = net->Layers[l];
        const float *in = (l == 0) ? m->x : m->a[l-1];
        for (int j = 0; j < c->nb_neuronnes; j++) {
            float dl = c->taux * m->delta[l][j];
            for (int i = 0; i < c->dim; i++) c->poids[j][i] -= dl * in[i];
            c->b[j] -= dl;
        }
    }

    // embeddings : chaque position k modifie la ligne de son caractère
    float lr = c0->taux;
    for (int k = 0; k < N; k++)
        for (int j = 0; j < d; j++)
            m->voc->emb[ctx[k]][j] -= lr * m->dx[k*d + j];
}

void train_modele(Modele *m, const Texte *train, const Texte *val, int epoques, float taux0){
    int N = m->N;      // Taille du ctx
    int ni = train->n - N; // Taille du texte - taille ctx -> nombre d'incrementation possible avant de depasser
    int *pos = malloc(ni * sizeof(int));
    for (int i = 0; i < ni; i++) pos[i] = N + i;

    for (int e = 0; e < epoques; e++) {
        float lr = taux0 / (1.0f + 3.0f * e / epoques);   // même décroissance que ton classifieur
        for (int l = 0; l < m->net->nb_layer; l++) m->net->Layers[l]->taux = lr;

        for (int i = ni - 1; i > 0; i--) {                // Fisher-Yates sur les positions
            int j = rand() % (i + 1);
            int tmp = pos[i]; pos[i] = pos[j]; pos[j] = tmp;
        }
        for (int i = 0; i < ni; i++) {
            int t = pos[i];
            update_modele(m, train->tok + t - N, train->tok[t]);
        }

        float Lt = calc_perte(m, train, 7);               // un exemple sur 7 suffit pour surveiller
        float Lv = calc_perte(m, val, 1);
        printf("epoque %d lr=%.4f perte train=%.4f val=%.4f perplexite val=%.2f\n",e + 1, lr, Lt, Lv, expf(Lv));
        fflush(stdout);
    }
    free(pos);
}