/*
#include "Rasterizer.h"

#include "Occluder.h"

#include <algorithm>
#include <cassert>
#include <cmath>
*/

namespace SoftwareRasterizer;

public enum PrimitiveMode : byte
{
  Culled = 0,
  Triangle0,
  Triangle1,
  ConcaveRight,
  ConcaveLeft,
  ConcaveCenter,
  Convex
};
