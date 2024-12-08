/*
 *            UVaFTLE 1.0: Lagrangian finite time 
 *		    Lyapunov exponent extraction 
 *		    for fluid dynamic applications
 *
 *    Copyright (C) 2023, 2024 Rocío Carratalá-Sáez et. al.
 *    This file is part of the UVaFTLE application.
 *
 *  UVaFTLE is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  UVaFTLE is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with UVaFTLE.  If not, see <http://www.gnu.org/licenses/>.
 */ 

#include <stdio.h>

#include "preprocess.h"
#include "arithmetic.h"

using namespace sycl;

void read_coordinates ( char *filename, int nDim, int nPoints, double *coords )
{
	int ip, d, check_EOF;
	char buffer[255];
	FILE *file;

	// Open file
	file = fopen( filename, "r" );

	// First element must be nPoints
	check_EOF = fscanf(file, "%s", buffer);
	if ( check_EOF == EOF )
	{
		fprintf( stderr, "Error: Unexpected EOF in read_coordinates\n" );
		exit(-1);
	}

	// Rest of read elements will be points' coordinates
	for ( ip = 0; ip < nPoints; ip++ )
	{
		for ( d = 0; d < nDim; d++ )
		{
			check_EOF = fscanf(file, "%s", buffer);
			if ( check_EOF == EOF )
			{
				fprintf( stderr, "Error: Unexpected EOF in read_coordinates\n" );
				exit(-1);
			}
			coords[ip * nDim + d] = atof(buffer);
		}
	}

	// Close file
	fclose(file);
}

void read_faces ( char *filename, int nDim, int nVertsPerFace, int nFaces, int *faces )
{
   int iface, ielem, check_EOF;
   char buffer[255];
   FILE *file;

   // Open file
   file = fopen( filename, "r" );

   // First element must be nFaces
   check_EOF = fscanf(file, "%s", buffer);
   if ( check_EOF == EOF )
   {
      fprintf( stderr, "Error: Unexpected EOF in read_faces\n" );
      exit(-1);
   }

   // Rest of read elements will be faces points' indices
   for ( iface = 0; iface < nFaces; iface++ )
   {
      for ( ielem = 0; ielem < nVertsPerFace; ielem++ )
      {
	 check_EOF = fscanf(file, "%s", buffer);
	 if ( check_EOF == EOF )
	 {
	    fprintf( stderr, "Error: Unexpected EOF in read_faces\n" );
	    exit(-1);
	 }
	 faces[iface * nVertsPerFace + ielem] = atoi(buffer);
      }
   }

   // Close file
   fclose(file);
}

void read_flowmap ( char *filename, int nDims, int nPoints, double *flowmap )
{
   int ip, idim, check_EOF;
   char buffer[255];
   FILE *file;

   // Open file
   file = fopen( filename, "r" );

   // Set velocity vectors space
   for ( ip = 0; ip < nPoints; ip++ )
   {
      for ( idim = 0; idim < nDims; idim++ )
	  {
		check_EOF = fscanf(file, "%s", buffer);
		if ( check_EOF == EOF )
		{
			fprintf( stderr, "Error: Unexpected EOF in read_flowmap\n" );
			exit(-1);
		}
		flowmap[ip * nDims + idim] = atof(buffer);
	  }
   }

   // Close file
   fclose(file);
}

void create_nFacesPerPoint_vector ( int nDim, int nPoints, int nFaces, int nVertsPerFace, int *faces, int *nFacesPerPoint )
{
	int ip, iface, ipf;
	for ( ip = 0; ip < nPoints; ip++ )
	{
		nFacesPerPoint[ip] = 0;
	}
	for ( iface = 0; iface < nFaces; iface++ )
	{
		for ( ipf = 0; ipf < nVertsPerFace; ipf++ )
		{
			ip = faces[iface * nVertsPerFace + ipf];
			nFacesPerPoint[ip] = nFacesPerPoint[ip] + 1;
		}
	}
	for ( ip = 1; ip < nPoints; ip++ )
	{
		nFacesPerPoint[ip] = nFacesPerPoint[ip] + nFacesPerPoint[ip-1];
	}	
}

event create_facesPerPoint_vector(queue* q, int nDim, int nPoints, int offset, int faces_offset, int nFaces, int nVertsPerFace, buffer<int, 1> *b_faces, buffer<int, 1> *b_nFacesPerPoint, buffer<int, 1> *b_facesPerPoint)
{
    // Create host-accessible buffers for start indices and write indices
    buffer<int, 1> b_startIndices(static_cast<size_t>(nPoints));
    buffer<int, 1> b_writeIndices(static_cast<size_t>(nPoints));

    // Zero-initialize write indices
    {
	host_accessor h_writeIndices{b_writeIndices, read_write};
        std::fill(h_writeIndices.begin(), h_writeIndices.end(), 0);
    }

    // Preliminary kernel to calculate start indices
    auto prelim_event = q->submit([&](handler &h) {
        accessor nFacesPerPoint{*b_nFacesPerPoint, h, read_only};
        accessor startIndices{b_startIndices, h, write_only, no_init};

        h.parallel_for(range<1>(static_cast<size_t>(nPoints)), [=](id<1> i) {
            int th_id = i[0];

            // Calculate start index for this point's faces
            startIndices[th_id] = (th_id == 0)
                ? 0
                : nFacesPerPoint[th_id-1] - faces_offset;
        });
    });

    return q->submit([&](handler &h) {
        accessor faces{*b_faces, h, read_only};
        accessor startIndices{b_startIndices, h, read_only};
        accessor writeIndices{b_writeIndices, h, read_write};
        accessor facesPerPoint{*b_facesPerPoint, h, write_only, no_init};

	h.depends_on(prelim_event);

        h.parallel_for<class optimized_preprocess>(
            nd_range<1>(
                range<1>{static_cast<size_t>((nFaces + BLOCK - 1) / BLOCK * BLOCK)},
                range<1>{static_cast<size_t>(BLOCK)}
            ),
            [=](nd_item<1> item) {
                // Equivalent to CUDA's blockIdx.x * blockDim.x + threadIdx.x
                int iface = item.get_global_id(0);

                // Early exit for out-of-bounds faces
                if (iface >= nFaces) return;

                // Iterate through vertices of the current face
                for (int ipf = 0; ipf < nVertsPerFace; ipf++) {
                    // Get point index for this vertex
                    int point = faces[iface * nVertsPerFace + ipf];

		    // Get atomic acccess for point in write indices
		    atomic_ref<int, memory_order::relaxed, memory_scope::device, access::address_space::global_space> atomic_point(writeIndices[point]);

                    // Atomic increment of write indices (similar to CUDA's atomicAdd)
                    int offset = atomic_point.fetch_add(1);

                    // Write face index to the computed position
                    facesPerPoint[startIndices[point] + offset] = iface;
                }
            }
        );
    });
}

