/*******************************************************************************
  Copyright (c) 2009, Limbic Software, Inc.
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:
      * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
      * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
      * Neither the name of the Limbic Software, Inc. nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY LIMBIC SOFTWARE, INC. ''AS IS'' AND ANY
  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL LIMBIC SOFTWARE, INC. BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ******************************************************************************/
#include "pvr.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

unsigned int countBits(unsigned int x)
{
    x  = x - ((x >> 1) & 0x55555555);
    x  = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    x  = x + (x >> 4);
    x &= 0xF0F0F0F;
    return (x * 0x01010101) >> 24;
}

typedef struct
{
    uint32_t PackedData[2];
}AMTC_BLOCK_STRUCT;

const unsigned int PVRTEX_CUBEMAP               = (1<<12);

extern void Decompress(AMTC_BLOCK_STRUCT *pCompressedData,
                       const int Do2bitMode,
                       const int XDim,
                       const int YDim,
                       const int AssumeImageTiles,
                       unsigned char* pResultImage);

/*******************************************************************************
  This PVR code is loosely based on Wolfgang Engel's Oolong Engine:

        http://oolongengine.com/

    Thank you, Wolfgang!
 ******************************************************************************/

const char *typeStrings[] =
{
    "<invalid>", "<invalid>", "<invalid>", "<invalid>",
    "<invalid>", "<invalid>", "<invalid>", "<invalid>",
    "<invalid>", "<invalid>", "<invalid>", "<invalid>",
    "<invalid>", "<invalid>", "<invalid>", "<invalid>",
    "RGBA4444", "RGBA5551", "RGBA8888", "RGB565",
    "RGB555", "RGB888", "I8", "AI8",
    "PVRTC2", "PVRTC4"
};

typedef struct PVRHeader
{
    uint32_t      size;
    uint32_t      height;
    uint32_t      width;
    uint32_t      mipcount;
    uint32_t      flags;
    uint32_t      texdatasize;
    uint32_t      bpp;
    uint32_t      rmask;
    uint32_t      gmask;
    uint32_t      bmask;
    uint32_t      amask;
    uint32_t      magic;
    uint32_t      numtex;
} PVRHeader;

PVRTexture::PVRTexture()
:data(NULL)
{
}

PVRTexture::~PVRTexture()
{
    if(this->data)
        free(this->data);
}

const char* PVRTexture::loadApplePVRTC(uint8_t* data, int size)
{
    // additional heuristic
    if(size>sizeof(PVRHeader))
    {
        PVRHeader *header = (PVRHeader *)data;
        if (header->size == sizeof( PVRHeader ) &&( header->magic == 0x21525650 ) )
            // this looks more like a PowerVR file.
            return "Magic number matches PowerVR not ApplePVRTC";
    }

    // default to 2bpp, 8x8
    int mode = 1;
    int res = 8;

    // this is a tough one, could be 2bpp 8x8, 4bpp 8x8
    if(size==32)
    {
        // assume 4bpp, 8x8
        mode = 0;
        res = 8;
    } else
    {
        // Detect if it's 2bpp or 4bpp
        int shift = 0;
        int test2bpp = 0x40; // 16x16
        int test4bpp = 0x80; // 16x16

        while(shift<10)
        {
            int s2 = shift<<1;

            if((test2bpp<<s2)&size)
            {
                res = 16<<shift;
                mode = 1;
                this->format = "PVRTC2";
                break;
            }

            if((test4bpp<<s2)&size)
            {
                res = 16<<shift;
                mode = 0;
                this->format = "PVRTC4";
                break;
            }


            ++shift;
        }

        if(shift==10)
            // no mode could be found.
            return "No mode could be found";
        printf("detected apple %ix%i %i bpp pvrtc\n", res, res, mode*2+2);
    }

    // there is no reliable way to know if it's a 2bpp or 4bpp file. Assuming
    this->width = res;
    this->height = res;
    this->bpp = (mode+1)*2;
    this->numMips = 0;
    this->data = (uint8_t*)malloc(this->width*this->height*4);

    Decompress((AMTC_BLOCK_STRUCT*)data, mode, this->width,
                    this->height, 0, this->data);

    for(int y=0; y<res/2; ++y)
    for(int x=0; x<res; ++x)
    {
        int src = (x+y*res)*4;
        int dst = (x+(res-y-1)*res)*4;

        for(int c=0; c<4; ++c)
        {
            uint8_t tmp = this->data[src+c];
            this->data[src+c] = this->data[dst+c];
            this->data[dst+c] = tmp;
        }
    }

    return 0;
}

ePVRLoadResult PVRTexture::load(const char *const path)
{
    uint8_t *data;
    unsigned int length;

    FILE *fp = fopen(path, "rb");
    if(fp==NULL)
        return PVR_LOAD_FILE_NOT_FOUND;

    fseek(fp, 0, SEEK_END);
    length = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    data = (uint8_t*)malloc(length);
    fread(data, 1, length, fp);

    fclose(fp);

    // use a heuristic to detect potential apple PVRTC formats
    if(countBits(length)==1)
    {
        // very likely to be apple PVRTC
        if(loadApplePVRTC(data, length))
            return PVR_LOAD_OKAY;
    }

    if(length<sizeof(PVRHeader))
    {
        free(data);
        return PVR_LOAD_INVALID_FILE;
    }

    // parse the header
    uint8_t* p = data;
    PVRHeader *header = (PVRHeader *)p;
    p += sizeof( PVRHeader );

    if( header->size != sizeof( PVRHeader ) )
    {
        free( data );
        return PVR_LOAD_INVALID_FILE;
    }

    if( header->magic != 0x21525650 )
    {
        free( data );
        return PVR_LOAD_INVALID_FILE;
    }

    if(header->numtex<1)
    {
        header->numtex = (header->flags & PVRTEX_CUBEMAP)?6:1;
    }

    if( header->numtex != 1 )
    {
        free( data );
        return PVR_LOAD_MORE_THAN_ONE_SURFACE;
    }

    if(header->width*header->height*header->bpp/8 > length-sizeof(PVRHeader))
    {
        return PVR_LOAD_INVALID_FILE;
    }

    int ptype = header->flags & PVR_PIXELTYPE_MASK;
    printf("Pixeltype: 0x%02x\n", ptype);

    this->width = header->width;
    this->height = header->height;
    this->numMips = header->mipcount;
    this->bpp = header->bpp;

    printf("Width: %i\n", this->width);
    printf("Height: %i\n", this->height);

    this->data = (uint8_t*)malloc(this->width*this->height*4);

    if(ptype<PVR_MAX_TYPE)
        this->format = typeStrings[ptype];
    else
        this->format = "<unknown>";

    switch(ptype)
    {
    case PVR_TYPE_RGBA4444:
        {
            uint8_t *in  = p;
            uint8_t *out = this->data;
            for(int y=0; y<this->height; ++y)
            for(int x=0; x<this->width; ++x)
            {
                int v1 = *in++;
                int v2 = *in++;

                uint8_t a = (v1&0x0f)<<4;
                uint8_t b = (v1&0xf0);
                uint8_t g = (v2&0x0f)<<4;
                uint8_t r = (v2&0xf0);

                *out++ = r;
                *out++ = g;
                *out++ = b;
                *out++ = a;
            }
        }
        break;
    case PVR_TYPE_RGBA5551:
        {
            uint8_t *in  = p;
            uint8_t *out = this->data;
            for(int y=0; y<this->height; ++y)
            for(int x=0; x<this->width; ++x)
            {
                unsigned short v = *(unsigned short*)in;
                in += 2;

                uint8_t r = (v&0xf800)>>8;
                uint8_t g = (v&0x07c0)>>3;
                uint8_t b = (v&0x003e)<<2;
                uint8_t a = (v&0x0001)?255:0;

                *out++ = r;
                *out++ = g;
                *out++ = b;
                *out++ = a;
            }
        }
        break;
    case PVR_TYPE_RGBA8888:
        {
            uint8_t *in  = p;
            uint8_t *out = this->data;
            for(int y=0; y<this->height; ++y)
            for(int x=0; x<this->width; ++x)
            {
                *out++ = *in++;
                *out++ = *in++;
                *out++ = *in++;
                *out++ = *in++;
            }
        }
        break;
    case PVR_TYPE_RGB565:
        {
            uint8_t *in  = p;
            uint8_t *out = this->data;
            for(int y=0; y<this->height; ++y)
            for(int x=0; x<this->width; ++x)
            {
                short v = *(short*)in;
                in += 2;


                uint8_t b = (v&0x001f)<<3;
                uint8_t g = (v&0x07e0)>>3;
                uint8_t r = (v&0xf800)>>8;
                uint8_t a = 255;

                if(x==128&&y==128)
                {
                    printf("%04x\n", v);
                    printf("%i %i %i\n", r, g, b);
                }

                *out++ = r;
                *out++ = g;
                *out++ = b;
                *out++ = a;
            }
        }
        break;
    case PVR_TYPE_RGB555:
        {
            uint8_t *in  = p;
            uint8_t *out = this->data;
            for(int y=0; y<this->height; ++y)
            for(int x=0; x<this->width; ++x)
            {
                short v = *(short*)in;
                in += 2;

                uint8_t r = (v&0x001f)<<3;
                uint8_t g = (v&0x03e0)>>2;
                uint8_t b = (v&0x7c00)>>7;
                uint8_t a = 255;

                *out++ = r;
                *out++ = g;
                *out++ = b;
                *out++ = a;
            }
        }
        break;
    case PVR_TYPE_RGB888:
        {
            uint8_t *in  = p;
            uint8_t *out = this->data;
            for(int y=0; y<this->height; ++y)
            for(int x=0; x<this->width; ++x)
            {
                *out++ = *in++;
                *out++ = *in++;
                *out++ = *in++;
                *out++ = 255;
            }
        }
        break;
    case PVR_TYPE_I8:
        {
            uint8_t *in  = p;
            uint8_t *out = this->data;
            for(int y=0; y<this->height; ++y)
            for(int x=0; x<this->width; ++x)
            {
                int i = *in++;

                *out++ = i;
                *out++ = i;
                *out++ = i;
                *out++ = 255;
            }
        }
        break;
    case PVR_TYPE_AI8:
        {
            uint8_t *in  = p;
            uint8_t *out = this->data;
            for(int y=0; y<this->height; ++y)
            for(int x=0; x<this->width; ++x)
            {
                int i = *in++;
                int a = *in++;

                *out++ = i;
                *out++ = i;
                *out++ = i;
                *out++ = a;
            }
        }
        break;
    case PVR_TYPE_PVRTC2:
        {
            Decompress((AMTC_BLOCK_STRUCT*)p, 1, this->width,
                    this->height, 1, this->data);
        } break;
    case PVR_TYPE_PVRTC4:
        {
            Decompress((AMTC_BLOCK_STRUCT*)p, 0, this->width,
                    this->height, 1, this->data);
        } break;
    default:
        printf("unknown PVR type %i!\n", ptype);
        free(this->data);
        this->data = NULL;
        free(data);
        return PVR_LOAD_UNKNOWN_TYPE;
    }

    free(data);
    return PVR_LOAD_OKAY;
}
