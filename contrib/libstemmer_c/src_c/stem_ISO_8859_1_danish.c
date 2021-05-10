/* Generated by Snowball 2.1.0 - https://snowballstem.org/ */

#include "../runtime/header.h"

#ifdef __cplusplus
extern "C" {
#endif
extern int danish_ISO_8859_1_stem(struct SN_env * z);
#ifdef __cplusplus
}
#endif
static int r_undouble(struct SN_env * z);
static int r_other_suffix(struct SN_env * z);
static int r_consonant_pair(struct SN_env * z);
static int r_main_suffix(struct SN_env * z);
static int r_mark_regions(struct SN_env * z);
#ifdef __cplusplus
extern "C" {
#endif


extern struct SN_env * danish_ISO_8859_1_create_env(void);
extern void danish_ISO_8859_1_close_env(struct SN_env * z);


#ifdef __cplusplus
}
#endif
static const symbol s_0_0[3] = { 'h', 'e', 'd' };
static const symbol s_0_1[5] = { 'e', 't', 'h', 'e', 'd' };
static const symbol s_0_2[4] = { 'e', 'r', 'e', 'd' };
static const symbol s_0_3[1] = { 'e' };
static const symbol s_0_4[5] = { 'e', 'r', 'e', 'd', 'e' };
static const symbol s_0_5[4] = { 'e', 'n', 'd', 'e' };
static const symbol s_0_6[6] = { 'e', 'r', 'e', 'n', 'd', 'e' };
static const symbol s_0_7[3] = { 'e', 'n', 'e' };
static const symbol s_0_8[4] = { 'e', 'r', 'n', 'e' };
static const symbol s_0_9[3] = { 'e', 'r', 'e' };
static const symbol s_0_10[2] = { 'e', 'n' };
static const symbol s_0_11[5] = { 'h', 'e', 'd', 'e', 'n' };
static const symbol s_0_12[4] = { 'e', 'r', 'e', 'n' };
static const symbol s_0_13[2] = { 'e', 'r' };
static const symbol s_0_14[5] = { 'h', 'e', 'd', 'e', 'r' };
static const symbol s_0_15[4] = { 'e', 'r', 'e', 'r' };
static const symbol s_0_16[1] = { 's' };
static const symbol s_0_17[4] = { 'h', 'e', 'd', 's' };
static const symbol s_0_18[2] = { 'e', 's' };
static const symbol s_0_19[5] = { 'e', 'n', 'd', 'e', 's' };
static const symbol s_0_20[7] = { 'e', 'r', 'e', 'n', 'd', 'e', 's' };
static const symbol s_0_21[4] = { 'e', 'n', 'e', 's' };
static const symbol s_0_22[5] = { 'e', 'r', 'n', 'e', 's' };
static const symbol s_0_23[4] = { 'e', 'r', 'e', 's' };
static const symbol s_0_24[3] = { 'e', 'n', 's' };
static const symbol s_0_25[6] = { 'h', 'e', 'd', 'e', 'n', 's' };
static const symbol s_0_26[5] = { 'e', 'r', 'e', 'n', 's' };
static const symbol s_0_27[3] = { 'e', 'r', 's' };
static const symbol s_0_28[3] = { 'e', 't', 's' };
static const symbol s_0_29[5] = { 'e', 'r', 'e', 't', 's' };
static const symbol s_0_30[2] = { 'e', 't' };
static const symbol s_0_31[4] = { 'e', 'r', 'e', 't' };

static const struct among a_0[32] =
{
{ 3, s_0_0, -1, 1, 0},
{ 5, s_0_1, 0, 1, 0},
{ 4, s_0_2, -1, 1, 0},
{ 1, s_0_3, -1, 1, 0},
{ 5, s_0_4, 3, 1, 0},
{ 4, s_0_5, 3, 1, 0},
{ 6, s_0_6, 5, 1, 0},
{ 3, s_0_7, 3, 1, 0},
{ 4, s_0_8, 3, 1, 0},
{ 3, s_0_9, 3, 1, 0},
{ 2, s_0_10, -1, 1, 0},
{ 5, s_0_11, 10, 1, 0},
{ 4, s_0_12, 10, 1, 0},
{ 2, s_0_13, -1, 1, 0},
{ 5, s_0_14, 13, 1, 0},
{ 4, s_0_15, 13, 1, 0},
{ 1, s_0_16, -1, 2, 0},
{ 4, s_0_17, 16, 1, 0},
{ 2, s_0_18, 16, 1, 0},
{ 5, s_0_19, 18, 1, 0},
{ 7, s_0_20, 19, 1, 0},
{ 4, s_0_21, 18, 1, 0},
{ 5, s_0_22, 18, 1, 0},
{ 4, s_0_23, 18, 1, 0},
{ 3, s_0_24, 16, 1, 0},
{ 6, s_0_25, 24, 1, 0},
{ 5, s_0_26, 24, 1, 0},
{ 3, s_0_27, 16, 1, 0},
{ 3, s_0_28, 16, 1, 0},
{ 5, s_0_29, 28, 1, 0},
{ 2, s_0_30, -1, 1, 0},
{ 4, s_0_31, 30, 1, 0}
};

static const symbol s_1_0[2] = { 'g', 'd' };
static const symbol s_1_1[2] = { 'd', 't' };
static const symbol s_1_2[2] = { 'g', 't' };
static const symbol s_1_3[2] = { 'k', 't' };

static const struct among a_1[4] =
{
{ 2, s_1_0, -1, -1, 0},
{ 2, s_1_1, -1, -1, 0},
{ 2, s_1_2, -1, -1, 0},
{ 2, s_1_3, -1, -1, 0}
};

static const symbol s_2_0[2] = { 'i', 'g' };
static const symbol s_2_1[3] = { 'l', 'i', 'g' };
static const symbol s_2_2[4] = { 'e', 'l', 'i', 'g' };
static const symbol s_2_3[3] = { 'e', 'l', 's' };
static const symbol s_2_4[4] = { 'l', 0xF8, 's', 't' };

static const struct among a_2[5] =
{
{ 2, s_2_0, -1, 1, 0},
{ 3, s_2_1, 0, 1, 0},
{ 4, s_2_2, 1, 1, 0},
{ 3, s_2_3, -1, 1, 0},
{ 4, s_2_4, -1, 2, 0}
};

static const unsigned char g_c[] = { 119, 223, 119, 1 };

static const unsigned char g_v[] = { 17, 65, 16, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 48, 0, 128 };

static const unsigned char g_s_ending[] = { 239, 254, 42, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16 };

static const symbol s_0[] = { 's', 't' };
static const symbol s_1[] = { 'i', 'g' };
static const symbol s_2[] = { 'l', 0xF8, 's' };

static int r_mark_regions(struct SN_env * z) {
    z->I[1] = z->l;
    {   int c_test1 = z->c;
z->c = z->c + 3;
        if (z->c > z->l) return 0;
        z->I[0] = z->c;
        z->c = c_test1;
    }
    if (out_grouping(z, g_v, 97, 248, 1) < 0) return 0;
    {   
        int ret = in_grouping(z, g_v, 97, 248, 1);
        if (ret < 0) return 0;
        z->c += ret;
    }
    z->I[1] = z->c;
    
    if (!(z->I[1] < z->I[0])) goto lab0;
    z->I[1] = z->I[0];
lab0:
    return 1;
}

static int r_main_suffix(struct SN_env * z) {
    int among_var;

    {   int mlimit1;
        if (z->c < z->I[1]) return 0;
        mlimit1 = z->lb; z->lb = z->I[1];
        z->ket = z->c;
        if (z->c <= z->lb || z->p[z->c - 1] >> 5 != 3 || !((1851440 >> (z->p[z->c - 1] & 0x1f)) & 1)) { z->lb = mlimit1; return 0; }
        among_var = find_among_b(z, a_0, 32);
        if (!(among_var)) { z->lb = mlimit1; return 0; }
        z->bra = z->c;
        z->lb = mlimit1;
    }
    switch (among_var) {
        case 1:
            {   int ret = slice_del(z);
                if (ret < 0) return ret;
            }
            break;
        case 2:
            if (in_grouping_b(z, g_s_ending, 97, 229, 0)) return 0;
            {   int ret = slice_del(z);
                if (ret < 0) return ret;
            }
            break;
    }
    return 1;
}

static int r_consonant_pair(struct SN_env * z) {
    {   int m_test1 = z->l - z->c;

        {   int mlimit2;
            if (z->c < z->I[1]) return 0;
            mlimit2 = z->lb; z->lb = z->I[1];
            z->ket = z->c;
            if (z->c - 1 <= z->lb || (z->p[z->c - 1] != 100 && z->p[z->c - 1] != 116)) { z->lb = mlimit2; return 0; }
            if (!(find_among_b(z, a_1, 4))) { z->lb = mlimit2; return 0; }
            z->bra = z->c;
            z->lb = mlimit2;
        }
        z->c = z->l - m_test1;
    }
    if (z->c <= z->lb) return 0;
    z->c--;
    z->bra = z->c;
    {   int ret = slice_del(z);
        if (ret < 0) return ret;
    }
    return 1;
}

static int r_other_suffix(struct SN_env * z) {
    int among_var;
    {   int m1 = z->l - z->c; (void)m1;
        z->ket = z->c;
        if (!(eq_s_b(z, 2, s_0))) goto lab0;
        z->bra = z->c;
        if (!(eq_s_b(z, 2, s_1))) goto lab0;
        {   int ret = slice_del(z);
            if (ret < 0) return ret;
        }
    lab0:
        z->c = z->l - m1;
    }

    {   int mlimit2;
        if (z->c < z->I[1]) return 0;
        mlimit2 = z->lb; z->lb = z->I[1];
        z->ket = z->c;
        if (z->c - 1 <= z->lb || z->p[z->c - 1] >> 5 != 3 || !((1572992 >> (z->p[z->c - 1] & 0x1f)) & 1)) { z->lb = mlimit2; return 0; }
        among_var = find_among_b(z, a_2, 5);
        if (!(among_var)) { z->lb = mlimit2; return 0; }
        z->bra = z->c;
        z->lb = mlimit2;
    }
    switch (among_var) {
        case 1:
            {   int ret = slice_del(z);
                if (ret < 0) return ret;
            }
            {   int m3 = z->l - z->c; (void)m3;
                {   int ret = r_consonant_pair(z);
                    if (ret < 0) return ret;
                }
                z->c = z->l - m3;
            }
            break;
        case 2:
            {   int ret = slice_from_s(z, 3, s_2);
                if (ret < 0) return ret;
            }
            break;
    }
    return 1;
}

static int r_undouble(struct SN_env * z) {

    {   int mlimit1;
        if (z->c < z->I[1]) return 0;
        mlimit1 = z->lb; z->lb = z->I[1];
        z->ket = z->c;
        if (in_grouping_b(z, g_c, 98, 122, 0)) { z->lb = mlimit1; return 0; }
        z->bra = z->c;
        z->S[0] = slice_to(z, z->S[0]);
        if (z->S[0] == 0) return -1;
        z->lb = mlimit1;
    }
    if (!(eq_v_b(z, z->S[0]))) return 0;
    {   int ret = slice_del(z);
        if (ret < 0) return ret;
    }
    return 1;
}

extern int danish_ISO_8859_1_stem(struct SN_env * z) {
    {   int c1 = z->c;
        {   int ret = r_mark_regions(z);
            if (ret < 0) return ret;
        }
        z->c = c1;
    }
    z->lb = z->c; z->c = z->l;

    {   int m2 = z->l - z->c; (void)m2;
        {   int ret = r_main_suffix(z);
            if (ret < 0) return ret;
        }
        z->c = z->l - m2;
    }
    {   int m3 = z->l - z->c; (void)m3;
        {   int ret = r_consonant_pair(z);
            if (ret < 0) return ret;
        }
        z->c = z->l - m3;
    }
    {   int m4 = z->l - z->c; (void)m4;
        {   int ret = r_other_suffix(z);
            if (ret < 0) return ret;
        }
        z->c = z->l - m4;
    }
    {   int m5 = z->l - z->c; (void)m5;
        {   int ret = r_undouble(z);
            if (ret < 0) return ret;
        }
        z->c = z->l - m5;
    }
    z->c = z->lb;
    return 1;
}

extern struct SN_env * danish_ISO_8859_1_create_env(void) { return SN_create_env(1, 2); }

extern void danish_ISO_8859_1_close_env(struct SN_env * z) { SN_close_env(z, 1); }

