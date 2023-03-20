#include "arithmetic.h"

double max_solve_3rd_degree_eq ( double a, double b, double c, double d)
{
	double x1, x2, x3;
	double A   = b*b - 3*a*c;
	double B   = b*c - 9*a*d;
	double C   = c*c - 3*b*d;
	double del = B*B - 4*A*C;
	if ( A == B && A == 0 )
	{
		x1 = x2 = x3 = -b/(3*a);
	}
	else if(del==0)
	{
		x1 = -b/a+B/A;
		x2 = x3 = (-B/A)/2;
	}
	else if(del<0)
	{
		double sqA = cl::sycl::sqrt(A);
		double sq3 = cl::sycl::sqrt(3.0);
		double T   = (2*A*b-3*a*B) / (2*A*sqA);
		double _xt = cl::sycl::acos(T);
		double xt  = _xt/3;
		x1         = (-b-2*sqA*cl::sycl::cos(xt)) / (3*a);
		x2         = (-b+sqA*(cl::sycl::cos(xt)+sq3*cl::sycl::sin(xt)))/(3*a);
		x3         = (-b+sqA*(cl::sycl::cos(xt)-sq3*cl::sycl::sin(xt)))/(3*a);
	}
	double max = ( x1 > x2 ) ? x1 : x2;
	return ( max > x3 ) ? max : x3;
}

void compute_gradient_2D (queue* q, int nPoints, int offset, int nVertsPerFace, cl::sycl::buffer<double, 1> *b_coords, cl::sycl::buffer<double, 1> *b_flowmap, cl::sycl::buffer<int, 1> *b_faces, 
					cl::sycl::buffer<int, 1> *b_nFacesPerPoint, cl::sycl::buffer<int, 1> *b_facesPerPoint, cl::sycl::buffer<double, 1> *b_log_sqrt, double T ){
	
	q->submit([&](handler &h){
		auto coords = b_coords->get_access<access::mode::read>(h);
		auto flowmap = b_flowmap->get_access<access::mode::read>(h);
		auto faces = b_faces->get_access<access::mode::read>(h);
		auto nFacesPerPoint = b_nFacesPerPoint->get_access<access::mode::read>(h);
		auto facesPerPoint = b_facesPerPoint->get_access<access::mode::read>(h);
		auto log_sqrt = b_log_sqrt->get_access<access::mode::discard_write>(h);
#if (defined(CUDA_DEVICE) || defined(HIP_DEVICE))		
		int size = (nPoints% BLOCK) ? (nPoints/BLOCK+1)*BLOCK: nPoints;
		h.parallel_for<class ftle2D> (nd_range<1>(range<1>{static_cast<size_t>(size)},range<1>{static_cast<size_t>(BLOCK)}), [=](nd_item<1> i){
		int ip = i.get_global_id(0) + offset;
		if(i.get_global_id(0) < nPoints){
#else
		h.parallel_for<class ftle2D> (range<1>{static_cast<size_t>(nPoints)}, [=](id<1> i){
			int ip = i[0] + offset;
#endif		
			int nDim = 2; 
			int iface, nFaces, idxface, ivert, faces_offset;
			int closest_points_0 = -1;
			int closest_points_1 = -1;
			int closest_points_2 = -1;
			int closest_points_3 = -1;
			int count = 0;
			int ivertex;
			double denom_x, denom_y;
			double gra10, gra11, gra20, gra21;
			double ftle_matrix[4], d_W_ei[2];
			faces_offset = (offset == 0) ? 0 :  nFacesPerPoint[offset-1];
			nFaces  = (ip == 0) ? nFacesPerPoint[ip] : nFacesPerPoint[ip] - nFacesPerPoint[ip-1];
			/* Find 4 closest points */
			
			for ( iface = 0; (iface < nFaces) && (count < 4); iface++ )
			{
				idxface = (ip == 0) ? facesPerPoint[iface] : facesPerPoint[nFacesPerPoint[ip-1] + iface - faces_offset];
				for ( ivert = 0; (ivert < nVertsPerFace) && (count < 4); ivert++ )
				{
					ivertex = faces[idxface * nVertsPerFace + ivert];
					if ( ivertex != ip )
					{
			   		/* (i-1, j) */
						if ( (coords[ivertex * nDim + 1] == coords[ip * nDim + 1]) && (coords[ivertex * nDim] < coords[ip * nDim]) )
						{
							if (closest_points_0 == -1)
							{
								closest_points_0 = ivertex;
								count++;
							}
						}
						else
						{
			   			/* (i+1, j) */
							if ( (coords[ivertex * nDim + 1] == coords[ip * nDim + 1]) && (coords[ivertex * nDim] > coords[ip * nDim]) )
							{
								if (closest_points_1 == -1)
								{
									closest_points_1 = ivertex;
									count++;
								}
							}
							else
							{
			   				/* (i, j-1) */
								if ( (coords[ivertex * nDim] == coords[ip * nDim]) && (coords[ivertex * nDim + 1] < coords[ip * nDim + 1]) ) 
								{
									if (closest_points_2 == -1)
									{
										closest_points_2 = ivertex;
										count++;
									}
								}
								else
								{
			   					/* (i, j+1) */
									if ( (coords[ivertex * nDim] == coords[ip * nDim]) && (coords[ivertex * nDim + 1] > coords[ip * nDim + 1]) )
									{
										if (closest_points_3 == -1)
										{
											closest_points_3 = ivertex;
											count++;
										}
									}
								}
							}
						}			
					}
				}
			}
			if ( count == 4 )
			{
				/* NOTE: take care with denom_x and denom_y zero */
				denom_x = coords[ closest_points_1 * nDim ]    - coords[ closest_points_0 * nDim ]; 
				denom_y = coords[ closest_points_3 * nDim + 1] - coords[ closest_points_2 * nDim + 1];
				gra10 = ( flowmap[ closest_points_1 * nDim ]     - flowmap[ closest_points_0 * nDim ] )     / denom_x; 			
				gra20 = ( flowmap[ closest_points_3 * nDim ]     - flowmap[ closest_points_2 * nDim ] )     / denom_y;
				gra11 = ( flowmap[ closest_points_1 * nDim + 1 ] - flowmap[ closest_points_0 * nDim + 1 ] ) / denom_x;
				gra21 = ( flowmap[ closest_points_3 * nDim + 1 ] - flowmap[ closest_points_2 * nDim + 1 ] ) / denom_y;
			}
			else
			{
				if ( count == 3 )
				{
					// (i-1, j) and (i+1, j) 
					if ( ( closest_points_0 > -1 ) && ( closest_points_1 > -1 ) )
					{
						denom_x = coords[ closest_points_1 * nDim ]    - coords[ closest_points_0 * nDim ]; 
						gra10 = ( flowmap[ closest_points_1 * nDim ] - flowmap[ closest_points_0 * nDim ] ) / denom_x;					
						gra20 = 1; //flowmap [ ip * nDim ]; //??
						gra11 = ( flowmap[ closest_points_1 * nDim + 1 ] - flowmap[ closest_points_0 * nDim + 1 ] ) / denom_x;
						gra21 = 1;//flowmap [ ip * nDim + 1];   //??
					}
					// (i-1, j) and (i+1, j) 
					else
					{
						denom_y = coords[ closest_points_3 * nDim + 1] - coords[ closest_points_2 * nDim + 1];
					  	gra10 = 1; //flowmap [ ip * nDim ];//??
					 	gra20 = ( flowmap[ closest_points_3 * nDim ]     - flowmap[ closest_points_2 * nDim ] )     / denom_y;
						gra11 = 1;// flowmap [ ip * nDim +1];//??
						gra21 = ( flowmap[ closest_points_3 * nDim + 1 ] - flowmap[ closest_points_2 * nDim + 1 ] ) / denom_y;
					}
				}
				else
				{
				  gra10 = 1;//flowmap [ ip * nDim ];
				  gra20 = 1;//flowmap [ ip * nDim ];
				  gra11 = 1;//flowmap [ ip * nDim + 1];
				  gra21 = 1;//flowmap [ ip * nDim + 1];
			   }
			}
			ftle_matrix[0] = gra10 * gra10 + gra11 * gra11;
			ftle_matrix[1] = gra10 * gra20 + gra11 * gra21;
			ftle_matrix[2] = gra20 * gra10 + gra21 * gra11;
			ftle_matrix[3] = gra20 * gra20 + gra21 * gra21;

			gra10 = ftle_matrix[0];
			gra11 = ftle_matrix[1];
			gra20 = ftle_matrix[2];
			gra21 = ftle_matrix[3];

			ftle_matrix[0] = gra10 * gra10 + gra11 * gra11;
			ftle_matrix[1] = gra10 * gra20 + gra11 * gra21;
			ftle_matrix[2] = gra20 * gra10 + gra21 * gra11;
			ftle_matrix[3] = gra20 * gra20 + gra21 * gra21;

			double A10 = ftle_matrix[0];
			double A11 = ftle_matrix[1];
			double A20 = ftle_matrix[2];
			double A21 = ftle_matrix[3];

			double sq = cl::sycl::sqrt(A21 * A21 + A10 * A10 - 2 * (A10 * A21) + 4 * (A11 * A20));
			d_W_ei[0] = (A21 + A10 + sq) / 2;
			d_W_ei[1] = (A21 + A10 - sq) / 2; 

			//---------------- max---sqrt---log


			double max = d_W_ei[0];	 //d_w[ip*nDim];

			if (d_W_ei[1] > max ) max = d_W_ei[1];

			max = cl::sycl::sqrt(max);
			max = cl::sycl::log (max);
#if (defined(CUDA_DEVICE) || defined(HIP_DEVICE))					
			log_sqrt[i.get_global_id(0)] = max / T;				
		    }
#else
			log_sqrt[i[0]] = max / T;		
#endif		    	
		}); /*End parallel for*/
	}); /*End submit*/	
}

void compute_gradient_3D (queue* q,  int nPoints, int offset, int nVertsPerFace, cl::sycl::buffer<double, 1> *b_coords, cl::sycl::buffer<double, 1> *b_flowmap, cl::sycl::buffer<int, 1> *b_faces, 
					cl::sycl::buffer<int, 1> *b_nFacesPerPoint, cl::sycl::buffer<int, 1> *b_facesPerPoint, cl::sycl::buffer<double, 1> *b_log_sqrt, double T )
{
		q->submit([&](handler &h){
		auto coords = b_coords->get_access<access::mode::read, access::target::constant_buffer>(h);
		auto flowmap = b_flowmap->get_access<access::mode::read, access::target::constant_buffer>(h);
		auto faces = b_faces->get_access<access::mode::read, access::target::constant_buffer>(h);
		auto nFacesPerPoint = b_nFacesPerPoint->get_access<access::mode::read, access::target::constant_buffer>(h);
		auto facesPerPoint = b_facesPerPoint->get_access<access::mode::read, access::target::constant_buffer>(h);
		auto log_sqrt = b_log_sqrt->get_access<access::mode::discard_write, access::target::global_buffer>(h);
#if defined(CUDA_DEVICE) || defined(HIP_DEVICE)		
		int size = (nPoints% BLOCK) ? (nPoints/BLOCK+1)*BLOCK: nPoints;
		h.parallel_for<class ftle3D> (nd_range<1>(range<1>{static_cast<size_t>(size)},range<1>{static_cast<size_t>(BLOCK)}), [=](nd_item<1> i){
		int ip = i.get_global_id(0) + offset;
		if(i.get_global_id(0) < nPoints){
#else
		h.parallel_for<class ftle2D> (range<1>{static_cast<size_t>(nPoints)}, [=](id<1> i){
			int ip = i[0] + offset;
#endif		
			int nDim = 3; 
			int iface, nFaces, idxface, ivert, faces_offset;
			int closest_points_0 = -1;
			int closest_points_1 = -1;
			int closest_points_2 = -1;
			int closest_points_3 = -1;
			int closest_points_4 = -1;
			int closest_points_5 = -1;
			int count = 0;
			
			int ivertex;
			double denom_x, denom_y, denom_z;
			double ftle_matrix[9];
			double gra10, gra11, gra12, gra20, gra21, gra22, gra30, gra31, gra32;
			faces_offset = (offset == 0) ? 0 :  nFacesPerPoint[offset-1];
			nFaces  = (ip == 0) ? nFacesPerPoint[ip] : nFacesPerPoint[ip] - nFacesPerPoint[ip-1];

			/* Find 6 closest points */
			for ( iface = 0; (iface < nFaces) && (count < 6); iface++ )	{
				idxface = (ip == 0) ? facesPerPoint[iface] : facesPerPoint[nFacesPerPoint[ip-1] + iface - faces_offset];
				for ( ivert = 0; (ivert < nVertsPerFace) && (count < 6); ivert++ )
				{
					ivertex = faces[idxface * nVertsPerFace + ivert];
					if ( ivertex != ip ){
							/* (i-1, j, k) */
						if (   (coords[ivertex * nDim + 1] == coords[ip * nDim + 1])
							&& (coords[ivertex * nDim + 2] == coords[ip * nDim + 2]) 
							&& (coords[ivertex * nDim]     <  coords[ip * nDim]) ) {
							if (closest_points_0 == -1){
								closest_points_0 = ivertex;
								count++;
							}
						}
						else	{/* (i+1, j, k) */
							if ((coords[ivertex * nDim + 1] == coords[ip * nDim + 1]) 
								&& (coords[ivertex * nDim + 2] == coords[ip * nDim + 2]) 
								&& (coords[ivertex * nDim]     >  coords[ip * nDim]) )	{
								if (closest_points_1 == -1)
								{
									closest_points_1 = ivertex;
									count++;
								}
							}
							else	{/* (i, j-1, k) */
								if (   (coords[ivertex * nDim]     == coords[ip * nDim]) 
									&& (coords[ivertex * nDim + 2] == coords[ip * nDim + 2]) 
									&& (coords[ivertex * nDim + 1] <  coords[ip * nDim + 1]) ){
									if (closest_points_2 == -1) {
										closest_points_2 = ivertex;
										count++;
									}
								}
								else { /* (i, j+1, k) */
									if ( (coords[ivertex * nDim]     == coords[ip * nDim]) 
										&& (coords[ivertex * nDim + 2] == coords[ip * nDim + 2]) 
										&& (coords[ivertex * nDim + 1] >  coords[ip * nDim + 1]) ) {
										if (closest_points_3 == -1) {
											closest_points_3 = ivertex;
											count++;
										}
									}
									else { /* (i, j, k-1) */
										if (   (coords[ivertex * nDim]     == coords[ip * nDim]) 
											&& (coords[ivertex * nDim + 1] == coords[ip * nDim + 1]) 
											&& (coords[ivertex * nDim + 2] <  coords[ip * nDim + 2]) ){
											if (closest_points_4 == -1) {
												closest_points_4 = ivertex;
												count++;
											}
										}
										else { /* (i, j, k+1) */
											if (   (coords[ivertex * nDim]     == coords[ip * nDim]) 
												&& (coords[ivertex * nDim + 1] == coords[ip * nDim + 1]) 
												&& (coords[ivertex * nDim + 2] >  coords[ip * nDim + 2]) ) {
												if (closest_points_5 == -1)
												{
													closest_points_5 = ivertex;
													count++;
												}
											}
										}
									}
								}
							}
						}			
					}
				}
			}     
			if ( count == 6 ){
				/* NOTE: take care with denom_x and denom_y zero */
				denom_x = coords[ closest_points_1 * nDim ]    - coords[ closest_points_0 * nDim ]; 
				denom_y = coords[ closest_points_3 * nDim + 1] - coords[ closest_points_2 * nDim + 1];
				denom_z = coords[ closest_points_5 * nDim + 2] - coords[ closest_points_4 * nDim + 2];

				gra10 = ( flowmap[ closest_points_1 * nDim ] - flowmap[ closest_points_0 * nDim ] ) / denom_x; 			
				gra11 = ( flowmap[ closest_points_3 * nDim ] - flowmap[ closest_points_2 * nDim ] ) / denom_y;
				gra12 = ( flowmap[ closest_points_5 * nDim ] - flowmap[ closest_points_4 * nDim ] ) / denom_z;

				gra20 = ( flowmap[ closest_points_1 * nDim + 1] - flowmap[ closest_points_0 * nDim + 1] ) / denom_x; 			
				gra21 = ( flowmap[ closest_points_3 * nDim + 1] - flowmap[ closest_points_2 * nDim + 1] ) / denom_y;
				gra22 = ( flowmap[ closest_points_5 * nDim + 1] - flowmap[ closest_points_4 * nDim + 1] ) / denom_z;

				gra30 = ( flowmap[ closest_points_1 * nDim + 2] - flowmap[ closest_points_0 * nDim + 2] ) / denom_x; 			
				gra31 = ( flowmap[ closest_points_3 * nDim + 2] - flowmap[ closest_points_2 * nDim + 2] ) / denom_y;
				gra32 = ( flowmap[ closest_points_5 * nDim + 2] - flowmap[ closest_points_4 * nDim + 2] ) / denom_z;
			} else {
				gra10 = 1;
				gra11 = 1;
				gra12 = 1;
				gra20 = 1;
				gra21 = 1;
				gra22 = 1;
				gra30 = 1;
				gra31 = 1;
				gra32 = 1;
			}

				/* Tens */
			ftle_matrix[0] = gra10 * gra10 + gra20 * gra20 + gra30 * gra30;
			ftle_matrix[1] = gra10 * gra11 + gra20 * gra21 + gra30 * gra31;
			ftle_matrix[2] = gra10 * gra12 + gra20 * gra22 + gra30 * gra32;
			ftle_matrix[3] = ftle_matrix[1];
			ftle_matrix[4] = gra11 * gra11 + gra21 * gra21 + gra31 * gra31;
			ftle_matrix[5] = gra11 * gra12 + gra11 * gra22 + gra31 * gra32;
			ftle_matrix[6] = ftle_matrix[2];
			ftle_matrix[7] = ftle_matrix[5];
			ftle_matrix[8] = gra12 * gra12 + gra22 * gra22 + gra32 * gra32;

			// Store copy to later multiply by transpose 
			gra10 = ftle_matrix[0];
			gra11 = ftle_matrix[1];
			gra12 = ftle_matrix[2];
			gra20 = ftle_matrix[3];
			gra21 = ftle_matrix[4];
			gra22 = ftle_matrix[5];
			gra30 = ftle_matrix[6];
			gra31 = ftle_matrix[7];
			gra32 = ftle_matrix[8];

			// Matrix mult 
			double A10 = gra10 * gra10 + gra11 * gra11 + gra12 * gra12;
			double A11 = gra10 * gra20 + gra11 * gra21 + gra12 * gra22;
			double A12 = gra10 * gra30 + gra11 * gra31 + gra12 * gra32;
			double A20 = gra20 * gra10 + gra21 * gra11 + gra22 * gra12;
			double A21 = gra20 * gra20 + gra21 * gra21 + gra22 * gra22;
			double A22 = gra20 * gra30 + gra21 * gra31 + gra22 * gra32;
			double A30 = gra30 * gra10 + gra31 * gra11 + gra32 * gra12;
			double A31 = gra30 * gra20 + gra31 * gra21 + gra32 * gra22;
			double A32 = gra30 * gra30 + gra31 * gra31 + gra32 * gra32;

			double a = -1;
			double b = A10 + A21 + A32;
			double c = A12 * A30 + A22 * A31 + A11 * A20 - A10 * A21 - A10 * A32 - A21 * A32;
			double d = A10 * A21 * A32 + A11 * A22 * A30 + A12 * A20 * A31 - A10 * A22 * A31 - A11 * A20 * A32 - A12 * A21 * A30;
			double max = max_solve_3rd_degree_eq ( a, b, c, d );
			max = cl::sycl::sqrt(max);
			max = cl::sycl::log (max);
#if defined(CUDA_DEVICE) || defined(HIP_DEVICE)			
			log_sqrt[i.get_global_id(0)] = max / T;				
		    }
#else
			log_sqrt[i[0]] = max / T;
#endif
		}); /*End parallel for*/
	}); /*End submit*/	

}
