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
#ifndef PVR_H
#define PVR_H

#include <stdint.h>

enum ePVRPixelType
{
    PVR_PIXELTYPE_MASK  = 0xff,
    PVR_TYPE_RGBA4444   = 0x10,
    PVR_TYPE_RGBA5551   = 0x11,
    PVR_TYPE_RGBA8888   = 0x12,
    PVR_TYPE_RGB565     = 0x13,
    PVR_TYPE_RGB555     = 0x14,
    PVR_TYPE_RGB888     = 0x15,
    PVR_TYPE_I8         = 0x16,
    PVR_TYPE_AI8        = 0x17,
    PVR_TYPE_PVRTC2     = 0x18,
    PVR_TYPE_PVRTC4     = 0x19,

    PVR_MAX_TYPE        = 0x20,
};

enum ePVRLoadResult
{
    PVR_LOAD_OKAY,
    PVR_LOAD_INVALID_FILE,
    PVR_LOAD_MORE_THAN_ONE_SURFACE,
    PVR_LOAD_FILE_NOT_FOUND,
    PVR_LOAD_UNKNOWN_TYPE,
    PVR_LOAD_UNKNOWN_ERROR,
};

struct PVRTexture
{
    PVRTexture();
    ~PVRTexture();
    ePVRLoadResult load(const char *const path);

    bool loadApplePVRTC(uint8_t* data, int size);

    unsigned int width;
    unsigned int height;
    unsigned int numMips;
    unsigned int bpp;
    const char *format;

    uint8_t *data;
};

#endif
