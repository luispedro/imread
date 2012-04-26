/*
 * LZW decoding
 * Copyright (c) 2003 Fabrice Bellard
 * Copyright (c) 2006 Konstantin Shishkov.
 * Licensed under LGPL, see Licenses/LGPL for full license
 */
#define LZW_MAXBITS                 12
#define LZW_SIZTABLE                (1<<LZW_MAXBITS)
struct LZWState {
    unsigned char *pbuf, *ebuf;
    int bbits;
    unsigned int bbuf;

    int cursize;                ///< The current code size
    int curmask;
    int codesize;
    int clear_code;
    int end_code;
    int newcodes;               ///< First available code
    int top_slot;               ///< Highest code for current size
    int extra_slot;
    int slot;                   ///< Last read code
    int fc, oc;
    unsigned char *sp;
    unsigned char stack[LZW_SIZTABLE];
    unsigned char suffix[LZW_SIZTABLE];
    unsigned short prefix[LZW_SIZTABLE];
    int bs;                     ///< current buffer size for GIF
};

int lzw_decode_init(LZWState *s, int csize, unsigned char *buf, int buf_size);
int lzw_decode(LZWState *s, unsigned char *buf, int len);


static const unsigned short mask[17] =
{
    0x0000, 0x0001, 0x0003, 0x0007,
    0x000F, 0x001F, 0x003F, 0x007F,
    0x00FF, 0x01FF, 0x03FF, 0x07FF,
    0x0FFF, 0x1FFF, 0x3FFF, 0x7FFF, 0xFFFF
};

/* get one code from stream */
static int lzw_get_code(struct LZWState * s)
{
    int c;
        while (s->bbits < s->cursize) {
            s->bbuf = (s->bbuf << 8) | (*s->pbuf++);
            s->bbits += 8;
        }
        c = s->bbuf >> (s->bbits - s->cursize);
    s->bbits -= s->cursize;
    return c & s->curmask;
}


int lzw_decode_init(LZWState *p, int csize, unsigned char *buf, int buf_size)
{
    struct LZWState *s = (struct LZWState *)p;

    if(csize < 1 || csize > LZW_MAXBITS)
        return -1;
    /* read buffer */
    s->pbuf = buf;
    s->ebuf = s->pbuf + buf_size;
    s->bbuf = 0;
    s->bbits = 0;
    s->bs = 0;

    /* decoder */
    s->codesize = csize;
    s->cursize = s->codesize + 1;
    s->curmask = mask[s->cursize];
    s->top_slot = 1 << s->cursize;
    s->clear_code = 1 << s->codesize;
    s->end_code = s->clear_code + 1;
    s->slot = s->newcodes = s->clear_code + 2;
    s->oc = s->fc = -1;
    s->sp = s->stack;

    s->extra_slot = 1;
    return 0;
}

/**
 * Decode given number of bytes
 * NOTE: the algorithm here is inspired from the LZW GIF decoder
 *  written by Steven A. Bennett in 1987.
 */
int lzw_decode(LZWState *p, unsigned char *buf, int len){
    int l, c, code, oc, fc;
    unsigned char *sp;
    struct LZWState *s = (struct LZWState *)p;

    if (s->end_code < 0)
        return 0;

    l = len;
    sp = s->sp;
    oc = s->oc;
    fc = s->fc;

    for (;;) {
        while (sp > s->stack) {
            *buf++ = *(--sp);
            if ((--l) == 0)
                goto the_end;
        }
        c = lzw_get_code(s);
        if (c == s->end_code) {
            break;
        } else if (c == s->clear_code) {
            s->cursize = s->codesize + 1;
            s->curmask = mask[s->cursize];
            s->slot = s->newcodes;
            s->top_slot = 1 << s->cursize;
            fc= oc= -1;
        } else {
            code = c;
            if (code == s->slot && fc>=0) {
                *sp++ = fc;
                code = oc;
            }else if(code >= s->slot)
                break;
            while (code >= s->newcodes) {
                *sp++ = s->suffix[code];
                code = s->prefix[code];
            }
            *sp++ = code;
            if (s->slot < s->top_slot && oc>=0) {
                s->suffix[s->slot] = code;
                s->prefix[s->slot++] = oc;
            }
            fc = code;
            oc = c;
            if (s->slot >= s->top_slot - s->extra_slot) {
                if (s->cursize < LZW_MAXBITS) {
                    s->top_slot <<= 1;
                    s->curmask = mask[++s->cursize];
                }
            }
        }
    }
    s->end_code = -1;
  the_end:
    s->sp = sp;
    s->oc = oc;
    s->fc = fc;
    return len - l;
}

