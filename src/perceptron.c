#include "../include/perceptron.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

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

    m->nbEpoque=0;
    m->nbTraining=0;
    m->version=0;

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
    if (n == NULL) return;
    for(int i = 0;i<n->nb_layer;i++){
        free_layer(n->Layers[i]);
    }
    free(n->Layers);
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

        float scale = (act == ACT_SOFTMAX) ? sqrtf(1.0f / dim) : sqrtf(3.0f / dim);
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
    if (p == NULL) return;
    int nb_neuronnes = p->nb_neuronnes;
    for(int i = 0;i<nb_neuronnes;i++){
        free(p->poids[i]);
    }
    free(p->b);
    free(p->poids);
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
    return m->a[indice_couche];
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
        inputs = m->a[i];
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

void   update_modele(Modele *m, int *ctx, int cible) {
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

void train_modele(Modele *m, const Texte *train, const Texte *val, int epoques, float taux0, char *best_path){

    m->nbTraining=m->nbTraining+1;
    m->version=m->version+0.1;

    int N = m->N;      // Taille du ctx
    int ni = train->n - N; // Taille du texte - taille ctx -> nombre d'incrementation possible avant de depasser
    int *pos = malloc(ni * sizeof(int));
    for (int i = 0; i < ni; i++) pos[i] = N + i;

    float meilleure = 1e30f;
    int meilleure_epoque = -1;

    int epoques_initial = m->nbEpoque;
    int maxEpoques = epoques + m->nbEpoque;

    printf("%d epoques\n",epoques);

    for (int e = epoques_initial; e < maxEpoques; e++) {
        float lr = taux0 / (1.0f + 3.0f * e / maxEpoques);   // Decroissance du taux
        for (int l = 0; l < m->net->nb_layer; l++) m->net->Layers[l]->taux = lr;

        for (int i = ni - 1; i > 0; i--) {                // Fisher-Yates
            int j = rand() % (i + 1);
            int tmp = pos[i]; pos[i] = pos[j]; pos[j] = tmp;
        }
        for (int i = 0; i < ni; i++) {
            int t = pos[i];
            update_modele(m, train->tok + t - N, train->tok[t]);
        }

        float Lt = calc_perte(m, train, 1);
        float Lv = calc_perte(m, val, 1);
        printf("epoque %d / %d ; lr=%.4f perte train=%.4f val=%.4f perplexite val=%.2f\n",e,maxEpoques - 1, lr, Lt, Lv, expf(Lv));
        if(e%10==0){
            printf("Texte généré : ");
            generer(m,500,0.5,val->tok);
            printf("\n\n");
            fflush(stdout);
        }

        m->nbEpoque=m->nbEpoque+1;

        if(best_path != NULL && (Lv < meilleure || meilleure_epoque==-1)){
            meilleure = Lv;
            meilleure_epoque = e;
            sauver_modele(m, best_path);
        }
    }
    free(pos);
}

int sauver_modele(const Modele *m, char *chemin){
    if (chemin == NULL) {
        fprintf(stderr, "Erreur : chemin de sauvegarde invalide\n");
        return -1;
    }

    FILE *file = fopen(chemin, "wb");

    if (file == NULL) {
        perror("Erreur ouverture");
        return -1;
    }

    fprintf(file, "N : %d\n",m->N);
    fprintf(file, "nbEpoque : %d\n",m->nbEpoque);
    fprintf(file, "nbTraining : %d\n",m->nbTraining);
    fprintf(file, "version : %f\n",m->version);

    Reseau *net = m->net;
    fprintf(file, "nb_layer : %d\n",net->nb_layer);

    fprintf(file, "# Couche i : act;dim;taux;nb_neuronnes;(b1,b2,...,bn);({w1,...,wn},{w1,...,wn},...,{w1,...,wn})\n");

    for(int i = 0; i<net->nb_layer;i++){
        Couche *c = net->Layers[i];
        int dim = c->dim;

        fprintf(file, "Couche %d : %d;%d;%f;%d;(",i,c->act,c->dim,c->taux,c->nb_neuronnes);
        for(int i = 0; i<c->nb_neuronnes;i++){
            if(i==0) fprintf(file,"%f",c->b[i]);
            else fprintf(file,",%f",c->b[i]);
        }
        fprintf(file,");(");
        for(int i = 0; i<c->nb_neuronnes;i++){
            if(i==0) fprintf(file,"{");
            else fprintf(file,",{");
            
            for(int j = 0; j<dim; j++){
                if(j==0) fprintf(file,"%f",c->poids[i][j]);
                else fprintf(file,",%f",c->poids[i][j]);
            }

            fprintf(file,"}");
        }
        fprintf(file,")\n");
    }

    fprintf(file, "# Vocab\n");

    Vocab *voc = m->voc;

    fprintf(file, "d : %d\n",voc->d);
    fprintf(file, "V : %d\n",voc->V);

    fprintf(file, "emb : (");
    for(int i = 0; i<voc->V;i++){
        if(i==0) fprintf(file,"{");
        else fprintf(file,",{");
        
        for(int j = 0; j<voc->d; j++){
            float value = voc->emb[i][j];
            if(j==0) fprintf(file,"%f",value);
            else fprintf(file,",%f",value);
        }

        fprintf(file,"}");
    }
    fprintf(file,")\n");

    fprintf(file, "dic : (");
    for(int j = 0; j<voc->V; j++){
        int value = voc->dic[j];
        if(j==0) fprintf(file,"%d",value);
        else fprintf(file,",%d",value);
    }

    fprintf(file, ")\ndic_inv : (");
    for(int j = 0; j<voc->V; j++){
        char value = voc->dic_inv[j];
        if(j==0) fprintf(file,"'%c'",value);
        else fprintf(file,",'%c'",value);
    }
    fprintf(file,")\n");

    if (fclose(file) != 0) {
        perror("Erreur");
        return -1;
    }

    return 0;
}

static char *read_entire_file(const char *chemin, size_t *total_size) {
    FILE *f = fopen(chemin, "rb");
    if (f == NULL) return NULL;

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long size = ftell(f);
    if (size < 0) {
        fclose(f);
        return NULL;
    }
    rewind(f);

    char *buffer = malloc((size_t)size + 1U);
    if (buffer == NULL) {
        fclose(f);
        return NULL;
    }

    size_t read_count = fread(buffer, 1, (size_t)size, f);
    fclose(f);
    if (read_count != (size_t)size) {
        free(buffer);
        return NULL;
    }

    buffer[size] = '\0';
    if (total_size != NULL) *total_size = (size_t)size;
    return buffer;
}

Modele *charger_modele(const char *chemin){
    if (chemin == NULL) return NULL;

    size_t size = 0;
    char *buffer = read_entire_file(chemin, &size);
    if (buffer == NULL) return NULL;

    char *p = buffer;
    int N = 0, nbEpoque = 0, nbTraining = 0, nb_layer = 0, V = 0, d = 0;
    float version = 0.0f;

    char *tag = strstr(p, "N : ");
    if (tag == NULL || sscanf(tag + 4, "%d", &N) != 1) {
        free(buffer);
        return NULL;
    }
    p = strchr(tag, '\n');
    if (p == NULL) {
        free(buffer);
        return NULL;
    }
    p++;

    tag = strstr(p, "nbEpoque : ");
    if (tag == NULL || sscanf(tag + 11, "%d", &nbEpoque) != 1) {
        free(buffer);
        return NULL;
    }
    p = strchr(tag, '\n');
    if (p == NULL) {
        free(buffer);
        return NULL;
    }
    p++;

    tag = strstr(p, "nbTraining : ");
    if (tag == NULL || sscanf(tag + 13, "%d", &nbTraining) != 1) {
        free(buffer);
        return NULL;
    }
    p = strchr(tag, '\n');
    if (p == NULL) {
        free(buffer);
        return NULL;
    }
    p++;

    tag = strstr(p, "version : ");
    if (tag == NULL || sscanf(tag + 10, "%f", &version) != 1) {
        free(buffer);
        return NULL;
    }
    p = strchr(tag, '\n');
    if (p == NULL) {
        free(buffer);
        return NULL;
    }
    p++;

    tag = strstr(p, "nb_layer : ");
    if (tag == NULL || sscanf(tag + 11, "%d", &nb_layer) != 1) {
        free(buffer);
        return NULL;
    }
    p = strchr(tag, '\n');
    if (p == NULL) {
        free(buffer);
        return NULL;
    }
    p++;

    while (*p == '#') {
        p = strchr(p, '\n');
        if (p == NULL) {
            free(buffer);
            return NULL;
        }
        p++;
    }

    Reseau *net = malloc(sizeof(Reseau));
    if (net == NULL) {
        free(buffer);
        return NULL;
    }
    net->nb_layer = nb_layer;
    net->Layers = calloc((size_t)nb_layer, sizeof(Couche *));
    if (net->Layers == NULL) {
        free(net);
        free(buffer);
        return NULL;
    }

    for (int l = 0; l < nb_layer; l++) {
        char *layer_tag = strstr(p, "Couche ");
        if (layer_tag == NULL) {
            free_network(net);
            free(buffer);
            return NULL;
        }
        int expected = 0;
        if (sscanf(layer_tag, "Couche %d :", &expected) != 1 || expected != l) {
            free_network(net);
            free(buffer);
            return NULL;
        }

        char *line_end = strchr(layer_tag, '\n');
        if (line_end == NULL) {
            free_network(net);
            free(buffer);
            return NULL;
        }

        char *rest = strchr(layer_tag, ':');
        if (rest == NULL) {
            free_network(net);
            free(buffer);
            return NULL;
        }
        rest++;

        char *after = rest;
        long act = strtol(after, &after, 10);
        if (*after != ';') {
            free_network(net);
            free(buffer);
            return NULL;
        }
        after++;

        long dim = strtol(after, &after, 10);
        if (*after != ';') {
            free_network(net);
            free(buffer);
            return NULL;
        }
        after++;

        float taux = strtof(after, &after);
        if (*after != ';') {
            free_network(net);
            free(buffer);
            return NULL;
        }
        after++;

        long nb_neuronnes = strtol(after, &after, 10);
        if (*after != ';') {
            free_network(net);
            free(buffer);
            return NULL;
        }
        after++;

        Couche *c = malloc(sizeof(Couche));
        if (c == NULL) {
            free_network(net);
            free(buffer);
            return NULL;
        }

        c->act = (int)act;
        c->dim = (int)dim;
        c->taux = taux;
        c->nb_neuronnes = (int)nb_neuronnes;
        c->b = malloc(sizeof(float) * (size_t)c->nb_neuronnes);
        c->poids = malloc(sizeof(float *) * (size_t)c->nb_neuronnes);
        if (c->b == NULL || c->poids == NULL) {
            free(c->b);
            free(c->poids);
            free(c);
            free_network(net);
            free(buffer);
            return NULL;
        }

        if (*after != '(') {
            free(c->b);
            free(c->poids);
            free(c);
            free_network(net);
            free(buffer);
            return NULL;
        }
        after++;

        for (int j = 0; j < c->nb_neuronnes; j++) {
            c->b[j] = strtof(after, &after);
            if (j + 1 < c->nb_neuronnes) {
                while (*after == ' ' || *after == '\t' || *after == '\n' || *after == '\r') after++;
                if (*after != ',') {
                    free(c->b);
                    free(c->poids);
                    free(c);
                    free_network(net);
                    free(buffer);
                    return NULL;
                }
                after++;
            }
        }

        while (*after == ' ' || *after == '\t' || *after == '\n' || *after == '\r') after++;
        if (*after != ')') {
            free(c->b);
            free(c->poids);
            free(c);
            free_network(net);
            free(buffer);
            return NULL;
        }
        after++;

        while (*after == ' ' || *after == '\t' || *after == '\n' || *after == '\r') after++;
        if (*after != ';') {
            free(c->b);
            free(c->poids);
            free(c);
            free_network(net);
            free(buffer);
            return NULL;
        }
        after++;
        while (*after == ' ' || *after == '\t' || *after == '\n' || *after == '\r') after++;
        if (*after != '(') {
            free(c->b);
            free(c->poids);
            free(c);
            free_network(net);
            free(buffer);
            return NULL;
        }
        after++;

        for (int j = 0; j < c->nb_neuronnes; j++) {
            c->poids[j] = malloc(sizeof(float) * (size_t)c->dim);
            if (c->poids[j] == NULL) {
                for (int k = 0; k < j; k++) free(c->poids[k]);
                free(c->b);
                free(c->poids);
                free(c);
                free_network(net);
                free(buffer);
                return NULL;
            }

            while (*after == ' ' || *after == '\t' || *after == '\n' || *after == '\r') after++;
            if (*after != '{') {
                free(c->poids[j]);
                for (int k = 0; k < j; k++) free(c->poids[k]);
                free(c->b);
                free(c->poids);
                free(c);
                free_network(net);
                free(buffer);
                return NULL;
            }
            after++;

            for (int i = 0; i < c->dim; i++) {
                c->poids[j][i] = strtof(after, &after);
                if (i + 1 < c->dim) {
                    while (*after == ' ' || *after == '\t' || *after == '\n' || *after == '\r') after++;
                    if (*after != ',') {
                        free(c->poids[j]);
                        for (int k = 0; k < j; k++) free(c->poids[k]);
                        free(c->b);
                        free(c->poids);
                        free(c);
                        free_network(net);
                        free(buffer);
                        return NULL;
                    }
                    after++;
                }
            }

            while (*after == ' ' || *after == '\t' || *after == '\n' || *after == '\r') after++;
            if (*after != '}') {
                free(c->poids[j]);
                for (int k = 0; k < j; k++) free(c->poids[k]);
                free(c->b);
                free(c->poids);
                free(c);
                free_network(net);
                free(buffer);
                return NULL;
            }
            after++;

            if (j + 1 < c->nb_neuronnes) {
                while (*after == ' ' || *after == '\t' || *after == '\n' || *after == '\r') after++;
                if (*after != ',') {
                    free(c->poids[j]);
                    for (int k = 0; k < j; k++) free(c->poids[k]);
                    free(c->b);
                    free(c->poids);
                    free(c);
                    free_network(net);
                    free(buffer);
                    return NULL;
                }
                after++;
            }
        }

        while (*after == ' ' || *after == '\t' || *after == '\n' || *after == '\r') after++;
        if (*after != ')') {
            for (int j = 0; j < c->nb_neuronnes; j++) free(c->poids[j]);
            free(c->b);
            free(c->poids);
            free(c);
            free_network(net);
            free(buffer);
            return NULL;
        }
        after++;

        net->Layers[l] = c;
        p = line_end + 1;
    }

    tag = strstr(p, "# Vocab");
    if (tag == NULL) {
        free_network(net);
        free(buffer);
        return NULL;
    }
    p = strchr(tag, '\n');
    if (p == NULL) {
        free_network(net);
        free(buffer);
        return NULL;
    }
    p++;

    tag = strstr(p, "d : ");
    if (tag == NULL || sscanf(tag + 4, "%d", &d) != 1) {
        free_network(net);
        free(buffer);
        return NULL;
    }
    p = strchr(tag, '\n');
    if (p == NULL) {
        free_network(net);
        free(buffer);
        return NULL;
    }
    p++;

    tag = strstr(p, "V : ");
    if (tag == NULL || sscanf(tag + 4, "%d", &V) != 1) {
        free_network(net);
        free(buffer);
        return NULL;
    }
    p = strchr(tag, '\n');
    if (p == NULL) {
        free_network(net);
        free(buffer);
        return NULL;
    }
    p++;

    Vocab *voc = malloc(sizeof(Vocab));
    if (voc == NULL) {
        free_network(net);
        free(buffer);
        return NULL;
    }
    voc->V = V;
    voc->d = d;
    voc->dic = calloc(512, sizeof(int));
    voc->dic_inv = calloc((size_t)V, sizeof(unsigned char));
    voc->emb = malloc(sizeof(float *) * (size_t)V);
    if (voc->dic == NULL || voc->dic_inv == NULL || voc->emb == NULL) {
        free(voc->dic);
        free(voc->dic_inv);
        free(voc->emb);
        free(voc);
        free_network(net);
        free(buffer);
        return NULL;
    }

    for (int i = 0; i < 512; i++) voc->dic[i] = -1;

    tag = strstr(p, "emb : (");
    if (tag == NULL) {
        free_vocab(voc);
        free_network(net);
        free(buffer);
        return NULL;
    }
    p = tag + strlen("emb : (");
    for (int i = 0; i < V; i++) {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
        if (*p != '{') {
            free_vocab(voc);
            free_network(net);
            free(buffer);
            return NULL;
        }
        p++;
        voc->emb[i] = malloc(sizeof(float) * (size_t)d);
        if (voc->emb[i] == NULL) {
            free_vocab(voc);
            free_network(net);
            free(buffer);
            return NULL;
        }
        for (int j = 0; j < d; j++) {
            voc->emb[i][j] = strtof(p, &p);
            if (j + 1 < d) {
                while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
                if (*p != ',') {
                    free_vocab(voc);
                    free_network(net);
                    free(buffer);
                    return NULL;
                }
                p++;
            }
        }
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
        if (*p != '}') {
            free_vocab(voc);
            free_network(net);
            free(buffer);
            return NULL;
        }
        p++;
        if (i + 1 < V) {
            while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
            if (*p != ',') {
                free_vocab(voc);
                free_network(net);
                free(buffer);
                return NULL;
            }
            p++;
        }
    }

    tag = strstr(p, ")\ndic : (");
    if (tag == NULL) {
        free_vocab(voc);
        free_network(net);
        free(buffer);
        return NULL;
    }
    p = tag + strlen(")\ndic : (");
    for (int i = 0; i < V; i++) {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
        if (*p == ')') break;
        int value = (int)strtol(p, &p, 10);
        if (i + 1 < V) {
            while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
            if (*p != ',') {
                free_vocab(voc);
                free_network(net);
                free(buffer);
                return NULL;
            }
            p++;
        }
        if (value >= 0 && value < V) {
            voc->dic_inv[value] = (unsigned char)i;
            voc->dic[(unsigned char)i] = value;
        }
    }

    tag = strstr(p, ")\ndic_inv : (");
    if (tag == NULL) {
        free_vocab(voc);
        free_network(net);
        free(buffer);
        return NULL;
    }
    p = tag + strlen(")\ndic_inv : (");
    for (int i = 0; i < V; i++) {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
        if (*p == ')') break;
        if (*p != '\'') {
            free_vocab(voc);
            free_network(net);
            free(buffer);
            return NULL;
        }
        p++;
        char c = *p;
        if (*p == '\\') {
            p++;
            switch (*p) {
                case 'n': c = '\n'; break;
                case 'r': c = '\r'; break;
                case 't': c = '\t'; break;
                case '0': c = '\0'; break;
                default: c = *p; break;
            }
        }
        voc->dic_inv[i] = (unsigned char)c;
        voc->dic[(unsigned char)c] = i;
        p++;
        if (*p == '\'') p++;
        if (i + 1 < V) {
            while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
            if (*p != ',') {
                free_vocab(voc);
                free_network(net);
                free(buffer);
                return NULL;
            }
            p++;
        }
    }

    free(buffer);

    Modele *m = malloc(sizeof(Modele));
    if (m == NULL) {
        free_vocab(voc);
        free_network(net);
        return NULL;
    }

    m->N = N;
    m->voc = voc;
    m->net = net;
    m->nbEpoque = nbEpoque;
    m->nbTraining = nbTraining;
    m->version = version;

    m->x = malloc(sizeof(float) * (size_t)N * (size_t)voc->d);
    m->dx = malloc(sizeof(float) * (size_t)N * (size_t)voc->d);
    m->a = malloc(sizeof(float *) * (size_t)net->nb_layer);
    m->delta = malloc(sizeof(float *) * (size_t)net->nb_layer);
    if (m->x == NULL || m->dx == NULL || m->a == NULL || m->delta == NULL) {
        free(m->x);
        free(m->dx);
        free(m->a);
        free(m->delta);
        free_modele(m);
        free_vocab(voc);
        free_network(net);
        return NULL;
    }

    for (int i = 0; i < net->nb_layer; i++) {
        m->a[i] = malloc(sizeof(float) * (size_t)net->Layers[i]->nb_neuronnes);
        m->delta[i] = malloc(sizeof(float) * (size_t)net->Layers[i]->nb_neuronnes);
        if (m->a[i] == NULL || m->delta[i] == NULL) {
            for (int j = 0; j < i; j++) {
                free(m->a[j]);
                free(m->delta[j]);
            }
            free(m->a);
            free(m->delta);
            free(m->x);
            free(m->dx);
            free(m);
            free_vocab(voc);
            free_network(net);
            return NULL;
        }
    }

    return m;
}

void encoder_amorce(const Vocab *voc, const char *texte, int N, int *out){
    if (voc == NULL || texte == NULL || out == NULL || N <= 0) return;

    int pad = (voc->dic[(unsigned char)' '] >= 0) ? voc->dic[(unsigned char)' '] : 0;
    int len = (int)strlen(texte);
    
    for (int i = 0; i < N; i++) {
        int idx = len - N + i;
        int token = (idx >= 0) ? voc->dic[(unsigned char)texte[idx]] : pad;
        out[i] = (token >= 0) ? token : pad;
    }
}