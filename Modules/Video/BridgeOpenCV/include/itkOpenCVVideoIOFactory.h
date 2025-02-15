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
#ifndef itkOpenCVVideoIOFactory_h
#define itkOpenCVVideoIOFactory_h

#include "itkObjectFactoryBase.h"
#include "itkVideoIOBase.h"
#include "ITKVideoBridgeOpenCVExport.h"

namespace itk
{
/**
 *\class OpenCVVideoIOFactory
 * \brief Create instances of OpenCVVideoIO objects using an object factory.
 *
 * \ingroup ITKVideoBridgeOpenCV
 */
class ITKVideoBridgeOpenCV_EXPORT OpenCVVideoIOFactory : public ObjectFactoryBase
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(OpenCVVideoIOFactory);

  /** Standard class type aliases. */
  using Self = OpenCVVideoIOFactory;
  using Superclass = ObjectFactoryBase;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Class methods used to interface with the registered factories. */
  virtual const char *
  GetITKSourceVersion() const;

  virtual const char *
  GetDescription() const;

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(OpenCVVideoIOFactory, ObjectFactoryBase);

  /** Register one factory of this type  */
  static void
  RegisterOneFactory()
  {
    auto OpenCVFactory = OpenCVVideoIOFactory::New();

    ObjectFactoryBase::RegisterFactoryInternal(OpenCVFactory);
  }

protected:
  OpenCVVideoIOFactory();
  ~OpenCVVideoIOFactory();
};
} // end namespace itk

#endif
