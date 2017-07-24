// Python binding
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <math.h>
#include <iostream>

/*
 * @INPUT:
 *    im   : 2D occupancy grid map (char)(int8_t)
 *    x_im : physical x positions of of the grid map (im) cells  
 *    y_im : physical y positions of of the grid map (im) cells  
 *    vp   : 2 x nvp = occupied x,y positions from range sensor (in physical unit)
 *    xs   : physical x positions you want to evaluate "correlation"
 *    ys   : physical y positions you want to evaluate "correlation"
 *
 * @OUTPUT:
 *  cpr_out: map correlations
 */
std::vector<std::vector<double>> 
mapCorrelation( const std::vector<std::vector<int8_t>>& im,
                const std::vector<double>& x_im,
                const std::vector<double>& y_im,
                const std::vector<std::vector<double>>& vp,
                const std::vector<double>& xs,
                const std::vector<double>& ys )
{
  std::vector<std::vector<double>> cpr_out;  
  if( vp.size() < 2 || im.size() < 1 )
    return cpr_out;
  
  int nx = static_cast<int>(im.size());
  int ny = static_cast<int>(im[0].size());
  int nxs = static_cast<int>(xs.size());
  int nys = static_cast<int>(ys.size());
  int np = static_cast<int>(vp[0].size());
  double xmin = x_im.front();
  double ymin = y_im.front();
  double xmax = x_im.back();
  double ymax = y_im.back();
  double xresolution = (xmax-xmin)/(nx-1);
  double yresolution = (ymax-ymin)/(ny-1);
  
  // Initialize cpr_out  
  cpr_out.resize(nxs);
  for (int jx = 0; jx < nxs; ++jx)
    cpr_out[jx].resize(nys);
  
  // Main correlation
  for (int k = 0; k < np; ++k)
  {
    double x0 = vp[0][k];
    double y0 = vp[1][k];
    for (int jy = 0; jy < nys; ++jy)
    {
      double y1 = y0 + ys[jy];
      int iy = (int) std::round((y1-ymin)/yresolution);
      if ((iy < 0) || (iy >= ny)) continue;
      for (int jx = 0; jx < nxs; ++jx)
      {
		    double x1 = x0 + xs[jx];
		    int ix = (int) std::round((x1-xmin)/xresolution);
		    if ((ix < 0) || (ix >= nx)) continue;
		    cpr_out[jx][jy] += im[ix][iy];
      }
    }
  }
  
  return cpr_out;
}


/*
 * @INPUT:
 *    x0t  :   current center of lidar scan x 
 *    y0t  :   current center of lidar scan y 
 *    xis  :   the cells at the end of the lidar scan x values
 *    yis  :   the cells at the end of the lidar scan y values
 *
 * @OUTPUT:
 *    xyio : (x,y) indices of cells in ray (output array)
 */
std::vector<std::vector<double>>
getMapCellsFromRay( int x0t, int y0t,
                    const std::vector<int>& xis,
                    const std::vector<int>& yis )
{
  std::vector<std::vector<double>> xyio(2);
  int nPoints = static_cast<int>(std::min(xis.size(),yis.size()));

  for (int ii=0; ii<nPoints; ++ii)
  {
    int x0 = x0t;
    int y0 = y0t;
    
    int x1 = xis[ii];
    int y1 = yis[ii];    

    bool steep = std::abs(y1 - y0) > std::abs(x1 - x0);
    if(steep){
        int temp = x0;
        x0 = y0;
        y0 = temp;
        temp = x1;
        x1 = y1;
        y1 = temp;
    }
    if(x0 > x1){
        int temp = x0;
        x0 = x1;
        x1 = temp;
        temp = y0;
        y0 = y1;
        y1 = temp;
    }
    int deltax = x1 - x0;
    int deltay = std::abs(y1 - y0);
    float error = static_cast<float>(deltax) / 2.0f;
    int y = y0;
    int ystep;
    if(y0 < y1)
        ystep = 1;
    else
        ystep = -1;
    
    if(steep)
    {
      for(int x=x0; x<(x1); x++)
      {
        xyio[0].push_back(x);
        xyio[1].push_back(y);
        error = error - static_cast<float>(deltay);
        if(error < 0){
            y += ystep;
           error += static_cast<float>(deltax);
        }
      }
    }
    else
    {
      for(int x=x0; x<(x1); x++)
      {
        xyio[0].push_back(x);
        xyio[1].push_back(y);
        error = error - static_cast<float>(deltay);
        if(error < 0){
          y += ystep;
          error += static_cast<float>(deltax);
        }
      }
    }
  }
  return xyio;
}


// Python binding
PYBIND11_PLUGIN(MapUtils)
{
  pybind11::module m("MapUtils", "Occupancy Grid Mapping Utilities");
  
  m.def("mapCorrelation", &mapCorrelation, "A function which determines the correlation between a laser scan and an occupancy grid map",
  pybind11::arg("im"), pybind11::arg("x_im"), pybind11::arg("y_im"),
  pybind11::arg("vp"), pybind11::arg("xs"),   pybind11::arg("ys") );
  
  m.def("getMapCellsFromRay", &getMapCellsFromRay, "A function which determines the cell locations corresponding to a ray",
  pybind11::arg("x0t"), pybind11::arg("y0t"), pybind11::arg("xis"), pybind11::arg("yis") );

  return m.ptr();  
}


