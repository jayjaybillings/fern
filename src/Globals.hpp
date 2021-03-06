/**----------------------------------------------------------------------------
Copyright (c) 2015-, The University of Tennessee
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of fern nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author(s): Jay Jay Billings, Ben Brock, Andrew Belt, Dan Shyles, Mike Guidry
-----------------------------------------------------------------------------*/
#include "fern_math.h"
#include <memory>
#include <vector>

#ifndef Globals_cuh
#define Globals_cuh

struct Network;

/**	Only allocated on the device per FERN integration
*/
class Globals
{
	// FIXME! These should all be std::shared_ptr!

public:
	std::vector<fern_real> preFac; // [reactions]
	std::vector<fern_real> Flux; // [reactions]
	std::vector<fern_real> Fplus; // [totalFplus]
	std::vector<fern_real> Fminus; // [totalFminus]
	std::vector<fern_real> rate; // [reactions]
	std::vector<fern_real> massNum;
	std::vector<fern_real> X;
	std::vector<fern_real> Fdiff;
	std::vector<fern_real> Yzero;
	std::vector<fern_real> FplusSum;
	std::vector<fern_real> FminusSum;

public:
	Globals(const Network & network);
};

#endif
