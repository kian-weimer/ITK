/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkIsConvertible_h
#define itkIsConvertible_h

#include "itkMacro.h"

#if !defined(ITK_LEGACY_REMOVE)

#  include "itkMetaProgrammingLibrary.h"

namespace itk
{

/** \cond HIDE_META_PROGRAMMING */

namespace mpl
{
namespace Details
{

/** Helper root class for Meta-programming purpose.
 * This class provides two types that help build SFINAE based meta-programs.
 * \ingroup MetaProgrammingLibrary
 * \ingroup ITKCommon
 */
struct SfinaeTypes
{
  using TOne = char;
  struct TTwo
  {
    char arr__[2];
  };
};
} // namespace Details

/** Traits that emulates \c std::is_convertible<>.
 * \tparam TFrom type to convert from
 * \tparam TTo type to convert to
 * \return (in \c Value) whether \c TFrom objects can be converted into \c TTo
 * objects.
 * \warning This version does not support \c void, function pointers, nor arrays.
 * \author The definition provided follows the code snippet available in Andrei
 * Alexandrescu's <em>Modern C++ Design</em>.
 * \ingroup MetaProgrammingLibrary
 * \ingroup ITKCommon
 */
template <typename TFrom, typename TTo>
struct IsConvertible : private Details::SfinaeTypes
{
private:
  static TOne Test(TTo);
  static TTwo
  Test(...);
  static TFrom
  MakeT();

public:
  static constexpr bool Value = sizeof(Test(MakeT())) == sizeof(TOne);
};

} // end namespace mpl

// itk::IsConvertible has moved to itk::mpl.
// Expect itk::IsConvertible to be deprecated.
using mpl::IsConvertible;

/** \endcond */

} // end namespace itk

#else // ITK_LEGACY_REMOVE
#  error Use C++ 11 std::is_convertible directly
#endif

#endif // itkIsConvertible_h
