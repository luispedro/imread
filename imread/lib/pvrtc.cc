/*
   This file has been taken from the Oolong Engine and was only slightly
   modified.

    Oolong Engine for the iPhone / iPod touch
    Copyright (c) 2007-2008 Wolfgang Engel  http://code.google.com/p/oolongengine/
    
    This software is provided 'as-is', without any express or implied warranty.
    In no event will the authors be held liable for any damages arising from the use of this software.
    Permission is granted to anyone to use this software for any purpose, 
    including commercial applications, and to alter it and redistribute it freely, 
    subject to the following restrictions:
    
    1. The origin of this software must not be misrepresented; you must not
        claim that you wrote the original software. If you use this software in
        a product, an acknowledgment in the product documentation would be
        appreciated but is not required.
    2. Altered source versions must be plainly marked as such, and must not be
        misrepresented as being the original software.
    3. This notice may not be removed or altered from any source distribution.
*/
#include <stdint.h>
#include <string.h>
#include <assert.h>

typedef struct
{
	uint32_t PackedData[2];
}AMTC_BLOCK_STRUCT;

#define PT_INDEX 2

#define BLK_Y_SIZE 4
#define BLK_X_MAX 8 

#define BLK_X_2BPP 8
#define BLK_X_4BPP 4

#define _ASSERT(X) assert(X)
#define POWER_OF_2(X)  util_number_is_power_2(X)
#define CLAMP(X, lower, upper) (_MIN(_MAX((X),(lower)), (upper)))

#define _MIN(X,Y) (((X)<(Y))? (X):(Y))
#define _MAX(X,Y) (((X)>(Y))? (X):(Y))
#define WRAP_COORD(Val, Size) ((Val) & ((Size)-1))
#define LIMIT_COORD(Val, Size, AssumeImageTiles) \
    ((AssumeImageTiles)? WRAP_COORD((Val), (Size)): CLAMP((Val), 0, (Size)-1))



/******************************************************************************
 * Function Name: util_number_is_power_2
 *
 * Inputs       : input - A number.
 * Outputs      : -
 * Returns      : TRUE if the number is an integer power of two, else FALSE.
 * Globals Used : -
 *
 * Description  : Check that a number is an integer power of two, i.e.
 *                1, 2, 4, 8, ... etc.
 *                Returns FALSE for zero.
 * Pre-condition: -
 *****************************************************************************/
int util_number_is_power_2( unsigned  input )
{
  unsigned minus1;

  if( !input ) return 0;

  minus1 = input - 1;
  return ( (input | minus1) == (input ^ minus1) ) ? 1 : 0;
}

/***********************************************************/
/*
// Unpack5554Colour
//
// Given a block, extract the colour information and convert to 5554 formats
*/
/***********************************************************/

static void Unpack5554Colour(const AMTC_BLOCK_STRUCT *pBlock,
							 int   ABColours[2][4])
{
	uint32_t RawBits[2];

	int i;

	/*
	// Extract A and B
	*/
	RawBits[0] = pBlock->PackedData[1] & (0xFFFE); /*15 bits (shifted up by one)*/
	RawBits[1] = pBlock->PackedData[1] >> 16;	   /*16 bits*/

	/*
	//step through both colours
	*/
	for(i = 0; i < 2; i++)
	{
		/*
		// if completely opaque
		*/
		if(RawBits[i] & (1<<15))
		{
			/*
			// Extract R and G (both 5 bit)
			*/
			ABColours[i][0] = (RawBits[i] >> 10) & 0x1F;
			ABColours[i][1] = (RawBits[i] >>  5) & 0x1F;

			/*
			// The precision of Blue depends on  A or B. If A then we need to
			// replicate the top bit to get 5 bits in total
			*/
			ABColours[i][2] = RawBits[i] & 0x1F;
			if(i==0)
			{
				ABColours[0][2] |= ABColours[0][2] >> 4;
			}

			/*
			// set 4bit alpha fully on...
			*/
			ABColours[i][3] = 0xF;
		}
		/*
		// Else if colour has variable translucency
		*/
		else
		{
			/*
			// Extract R and G (both 4 bit).
			// (Leave a space on the end for the replication of bits
			*/
			ABColours[i][0] = (RawBits[i] >>  (8-1)) & 0x1E;
			ABColours[i][1] = (RawBits[i] >>  (4-1)) & 0x1E;

			/*
			// replicate bits to truly expand to 5 bits
			*/
			ABColours[i][0] |= ABColours[i][0] >> 4;
			ABColours[i][1] |= ABColours[i][1] >> 4;

			/*
			// grab the 3(+padding) or 4 bits of blue and add an extra padding bit
			*/
			ABColours[i][2] = (RawBits[i] & 0xF) << 1;

			/*
			// expand from 3 to 5 bits if this is from colour A, or 4 to 5 bits if from
			// colour B
			*/
			if(i==0)
			{
				ABColours[0][2] |= ABColours[0][2] >> 3;
			}
			else
			{
				ABColours[0][2] |= ABColours[0][2] >> 4;
			}

			/*
			// Set the alpha bits to be 3 + a zero on the end
			*/
			ABColours[i][3] = (RawBits[i] >> 11) & 0xE;
		}/*end if variable alpha*/
	}/*end for i*/

}


/***********************************************************/
/*
// UnpackModulations
//
// Given the block and the texture type and it's relative position in the
// 2x2 group of blocks, extract the bit patterns for the fully defined pixels.
*/
/***********************************************************/

static void	UnpackModulations(const AMTC_BLOCK_STRUCT *pBlock,
							  const int Do2bitMode,
							  int ModulationVals[8][16],
							  int ModulationModes[8][16],
							  int StartX,
							  int StartY)
{
	int BlockModMode;
	uint32_t ModulationBits;

	int x, y;

	BlockModMode= pBlock->PackedData[1] & 1;
	ModulationBits	= pBlock->PackedData[0];

	/*
	// if it's in an interpolated mode
	*/
	if(Do2bitMode && BlockModMode)
	{
		/*
		// run through all the pixels in the block. Note we can now treat all the
		// "stored" values as if they have 2bits (even when they didn't!)
		*/
		for(y = 0; y < BLK_Y_SIZE; y++)
		{
			for(x = 0; x < BLK_X_2BPP; x++)
			{
				ModulationModes[y+StartY][x+StartX] = BlockModMode;

				/*
				// if this is a stored value...
				*/
				if(((x^y)&1) == 0)
				{
					ModulationVals[y+StartY][x+StartX] = ModulationBits & 3;
					ModulationBits >>= 2;
				}
			}
		}/*end for y*/
	}
	/*
	// else if direct encoded 2bit mode - i.e. 1 mode bit per pixel
	*/
	else if(Do2bitMode)
	{
		for(y = 0; y < BLK_Y_SIZE; y++)
		{
			for(x = 0; x < BLK_X_2BPP; x++)
			{
				ModulationModes[y+StartY][x+StartX] = BlockModMode;

				/*
				// double the bits so 0=> 00, and 1=>11
				*/
				if(ModulationBits & 1)
				{
					ModulationVals[y+StartY][x+StartX] = 0x3;
				}
				else
				{
					ModulationVals[y+StartY][x+StartX] = 0x0;
				}
				ModulationBits >>= 1;
			}
		}/*end for y*/
	}
	/*
	// else its the 4bpp mode so each value has 2 bits
	*/
	else
	{
		for(y = 0; y < BLK_Y_SIZE; y++)
		{
			for(x = 0; x < BLK_X_4BPP; x++)
			{
				ModulationModes[y+StartY][x+StartX] = BlockModMode;

				ModulationVals[y+StartY][x+StartX] = ModulationBits & 3;
				ModulationBits >>= 2;
			}
		}/*end for y*/
	}

	/*
	// make sure nothing is left over
	*/
	_ASSERT(ModulationBits==0);
}


/***********************************************************/
/*
// Interpolate Colours
//
//
// This performs a HW bit accurate interpolation of either the
// A or B colours for a particular pixel
//
// NOTE: It is assumed that the source colours are in ARGB 5554 format -
//		 This means that some "preparation" of the values will be necessary.
*/
/***********************************************************/

static void InterpolateColours(const int ColourP[4],
						  const int ColourQ[4],
						  const int ColourR[4],
						  const int ColourS[4],
						  const int Do2bitMode,
						  const int x,
						  const int y,
						  int Result[4])
{
	int u, v, uscale;
	int k;

	int tmp1, tmp2;

	int P[4], Q[4], R[4], S[4];

	/*
	// Copy the colours
	*/
	for(k = 0; k < 4; k++)
	{
		P[k] = ColourP[k];
		Q[k] = ColourQ[k];
		R[k] = ColourR[k];
		S[k] = ColourS[k];
	}

	/*
	// put the x and y values into the right range
	*/
	v = (y & 0x3) | ((~y & 0x2) << 1);
	if(Do2bitMode)
	{
		u = (x & 0x7) | ((~x & 0x4) << 1);

	}
	else
	{
		u = (x & 0x3) | ((~x & 0x2) << 1);
	}



	/*
	// get the u and v scale amounts
	*/
	v  = v - BLK_Y_SIZE/2;

	if(Do2bitMode)
	{
		u = u - BLK_X_2BPP/2;
		uscale = 8;
	}
	else
	{
		u = u - BLK_X_4BPP/2;
		uscale = 4;
	}

	for(k = 0; k < 4; k++)
	{
		tmp1 = P[k] * uscale + u * (Q[k] - P[k]);
		tmp2 = R[k] * uscale + u * (S[k] - R[k]);

		tmp1 = tmp1 * 4 + v * (tmp2 - tmp1);

		Result[k] = tmp1;
	}

	/*
	// Lop off the appropriate number of bits to get us to 8 bit precision
	*/
	if(Do2bitMode)
	{
		/*
		// do RGB
		*/
		for(k = 0; k < 3; k++)
		{
			Result[k] >>= 2;
		}

		Result[3] >>= 1;
	}
	else
	{
		/*
		// do RGB  (A is ok)
		*/
		for(k = 0; k < 3; k++)
		{
			Result[k] >>= 1;
		}
	}

	/*
	// sanity check
	*/
	for(k = 0; k < 4; k++)
	{
		_ASSERT(Result[k] < 256);
	}


	/*
	// Convert from 5554 to 8888
	//
	// do RGB 5.3 => 8
	*/
	for(k = 0; k < 3; k++)
	{
		Result[k] += Result[k] >> 5;
	}
	Result[3] += Result[3] >> 4;

	/*
	// 2nd sanity check
	*/
	for(k = 0; k < 4; k++)
	{
		_ASSERT(Result[k] < 256);
	}

}

/***********************************************************/
/*
// GetModulationValue
//
// Get the modulation value as a numerator of a fraction of 8ths
*/
/***********************************************************/
static void GetModulationValue(int x,
							   int y,
							   const int Do2bitMode,
							   const int ModulationVals[8][16],
							   const int ModulationModes[8][16],
							   int *Mod,
							   int *DoPT)
{
	static const int RepVals0[4] = {0, 3, 5, 8};
	static const int RepVals1[4] = {0, 4, 4, 8};

	int ModVal;

	/*
	// Map X and Y into the local 2x2 block
	*/
	y = (y & 0x3) | ((~y & 0x2) << 1);
	if(Do2bitMode)
	{
		x = (x & 0x7) | ((~x & 0x4) << 1);

	}
	else
	{
		x = (x & 0x3) | ((~x & 0x2) << 1);
	}

	/*
	// assume no PT for now
	*/
	*DoPT = 0;

	/*
	// extract the modulation value. If a simple encoding
	*/
	if(ModulationModes[y][x]==0)
	{
		ModVal = RepVals0[ModulationVals[y][x]];
	}
	else if(Do2bitMode)
	{
		/*
		// if this is a stored value
		*/
		if(((x^y)&1)==0)
		{
			ModVal = RepVals0[ModulationVals[y][x]];
		}
		/*
		// else average from the neighbours
		//
		// if H&V interpolation...
		*/
		else if(ModulationModes[y][x] == 1)
		{
			ModVal = (RepVals0[ModulationVals[y-1][x]] +
					  RepVals0[ModulationVals[y+1][x]] +
					  RepVals0[ModulationVals[y][x-1]] +
					  RepVals0[ModulationVals[y][x+1]] + 2) / 4;
		}
		/*
		// else if H-Only
		*/
		else if(ModulationModes[y][x] == 2)
		{
			ModVal = (RepVals0[ModulationVals[y][x-1]] +
					  RepVals0[ModulationVals[y][x+1]] + 1) / 2;
		}
		/*
		// else it's V-Only
		*/
		else
		{
			ModVal = (RepVals0[ModulationVals[y-1][x]] +
					  RepVals0[ModulationVals[y+1][x]] + 1) / 2;

		}/*end if/then/else*/
	}
	/*
	// else it's 4BPP and PT encoding
	*/
	else
	{
		ModVal = RepVals1[ModulationVals[y][x]];

		*DoPT = ModulationVals[y][x] == PT_INDEX;
	}

	*Mod =ModVal;
}


/*****************************************************************************/
/*
// TwiddleUV
//
// Given the Block (or pixel) coordinates and the dimension of the texture
// in blocks (or pixels) this returns the twiddled offset of the block
// (or pixel) from the start of the map.
//
// NOTE the dimensions of the texture must be a power of 2
*/
/*****************************************************************************/

static int DisableTwiddlingRoutine = 0;

static uint32_t TwiddleUV(uint32_t YSize, uint32_t XSize, uint32_t YPos, uint32_t XPos)
{
	uint32_t Twiddled;

	uint32_t MinDimension;
	uint32_t MaxValue;

	uint32_t SrcBitPos;
	uint32_t DstBitPos;

	int ShiftCount;

	_ASSERT(YPos < YSize);
	_ASSERT(XPos < XSize);

	_ASSERT(POWER_OF_2(YSize));
	_ASSERT(POWER_OF_2(XSize));


	if(YSize < XSize)
	{
		MinDimension = YSize;
		MaxValue	 = XPos;
	}
	else
	{
		MinDimension = XSize;
		MaxValue	 = YPos;
	}

	/*
	// Nasty hack to disable twiddling
	*/
	if(DisableTwiddlingRoutine)
	{
		return (YPos* XSize + XPos);
	}

	/*
	// Step through all the bits in the "minimum" dimension
	*/
	SrcBitPos = 1;
	DstBitPos = 1;
	Twiddled  = 0;
	ShiftCount = 0;

	while(SrcBitPos < MinDimension)
	{
		if(YPos & SrcBitPos)
		{
			Twiddled |= DstBitPos;
		}

		if(XPos & SrcBitPos)
		{
			Twiddled |= (DstBitPos << 1);
		}


		SrcBitPos <<= 1;
		DstBitPos <<= 2;
		ShiftCount += 1;

	}/*end while*/

	/*
	// prepend any unused bits
	*/
	MaxValue >>= ShiftCount;

	Twiddled |=  (MaxValue << (2*ShiftCount));

	return Twiddled;
}

/***********************************************************/
/*
// Decompress
//
// Takes the compressed input data and outputs the equivalent decompressed
// image.
*/
/***********************************************************/

extern void Decompress(AMTC_BLOCK_STRUCT *pCompressedData,
				const int Do2bitMode,
				const int XDim,
				const int YDim,
				const int AssumeImageTiles,
				unsigned char* pResultImage)
{
	int x, y;
	int i, j;

	int BlkX, BlkY;
	int BlkXp1, BlkYp1;
	int XBlockSize;
	int BlkXDim, BlkYDim;

	int StartX, StartY;

	int ModulationVals[8][16];
	int ModulationModes[8][16];

	int Mod, DoPT;

	unsigned int uPosition;

	/*
	// local neighbourhood of blocks
	*/
	AMTC_BLOCK_STRUCT *pBlocks[2][2];

	AMTC_BLOCK_STRUCT *pPrevious[2][2] = {{NULL, NULL}, {NULL, NULL}};

	/*
	// Low precision colours extracted from the blocks
	*/
	struct
	{
		int Reps[2][4];
	}Colours5554[2][2];

	/*
	// Interpolated A and B colours for the pixel
	*/
	int ASig[4], BSig[4];

	int Result[4];

	if(Do2bitMode)
	{
		XBlockSize = BLK_X_2BPP;
	}
	else
	{
		XBlockSize = BLK_X_4BPP;
	}


	/*
	// For MBX don't allow the sizes to get too small
	*/
	BlkXDim = _MAX(2, XDim / XBlockSize);
	BlkYDim = _MAX(2, YDim / BLK_Y_SIZE);

	/*
	// Step through the pixels of the image decompressing each one in turn
	//
	// Note that this is a hideously inefficient way to do this!
	*/
	for(y = 0; y < YDim; y++)
	{
		for(x = 0; x < XDim; x++)
		{
			/*
			// map this pixel to the top left neighbourhood of blocks
			*/
			BlkX = (x - XBlockSize/2);
			BlkY = (y - BLK_Y_SIZE/2);

			BlkX = LIMIT_COORD(BlkX, XDim, AssumeImageTiles);
			BlkY = LIMIT_COORD(BlkY, YDim, AssumeImageTiles);


			BlkX /= XBlockSize;
			BlkY /= BLK_Y_SIZE;

			//BlkX = LIMIT_COORD(BlkX, BlkXDim, AssumeImageTiles);
			//BlkY = LIMIT_COORD(BlkY, BlkYDim, AssumeImageTiles);


			/*
			// compute the positions of the other 3 blocks
			*/
			BlkXp1 = LIMIT_COORD(BlkX+1, BlkXDim, AssumeImageTiles);
			BlkYp1 = LIMIT_COORD(BlkY+1, BlkYDim, AssumeImageTiles);

			/*
			// Map to block memory locations
			*/
			pBlocks[0][0] = pCompressedData +TwiddleUV(BlkYDim, BlkXDim, BlkY, BlkX);
			pBlocks[0][1] = pCompressedData +TwiddleUV(BlkYDim, BlkXDim, BlkY, BlkXp1);
			pBlocks[1][0] = pCompressedData +TwiddleUV(BlkYDim, BlkXDim, BlkYp1, BlkX);
			pBlocks[1][1] = pCompressedData +TwiddleUV(BlkYDim, BlkXDim, BlkYp1, BlkXp1);


			/*
			// extract the colours and the modulation information IF the previous values
			// have changed.
			*/
			if(memcmp(pPrevious, pBlocks, 4*sizeof(void*)) != 0)
			{
				StartY = 0;
				for(i = 0; i < 2; i++)
				{
					StartX = 0;
					for(j = 0; j < 2; j++)
					{
						Unpack5554Colour(pBlocks[i][j], Colours5554[i][j].Reps);

						UnpackModulations(pBlocks[i][j],
										  Do2bitMode,
										  ModulationVals,
										  ModulationModes,
										  StartX, StartY);

						StartX += XBlockSize;
					}/*end for j*/

					StartY += BLK_Y_SIZE;
				}/*end for i*/

				/*
				// make a copy of the new pointers
				*/
				memcpy(pPrevious, pBlocks, 4*sizeof(void*));
			}/*end if the blocks have changed*/


			/*
			// decompress the pixel.  First compute the interpolated A and B signals
			*/
			InterpolateColours(Colours5554[0][0].Reps[0],
							   Colours5554[0][1].Reps[0],
							   Colours5554[1][0].Reps[0],
							   Colours5554[1][1].Reps[0],
							   Do2bitMode, x, y,
							   ASig);

			InterpolateColours(Colours5554[0][0].Reps[1],
							   Colours5554[0][1].Reps[1],
							   Colours5554[1][0].Reps[1],
							   Colours5554[1][1].Reps[1],
							   Do2bitMode, x, y,
							   BSig);

			GetModulationValue(x,y, Do2bitMode, (const int (*)[16])ModulationVals, (const int (*)[16])ModulationModes,
							   &Mod, &DoPT);

			/*
			// compute the modulated colour
			*/
			for(i = 0; i < 4; i++)
			{
				Result[i] = ASig[i] * 8 + Mod * (BSig[i] - ASig[i]);
				Result[i] >>= 3;
			}
			if(DoPT)
			{
				Result[3] = 0;
			}

			/*
			// Store the result in the output image
			*/
			uPosition = (x+y*XDim)<<2;
			pResultImage[uPosition+0] = (uint8_t)Result[0];
			pResultImage[uPosition+1] = (uint8_t)Result[1];
			pResultImage[uPosition+2] = (uint8_t)Result[2];
			pResultImage[uPosition+3] = (uint8_t)Result[3];

		}/*end for x*/
	}/*end for y*/

}
