// Copyright 2016 Proyectos y Sistemas de Mantenimiento SL (eProsima).
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*!
 * @file metadata.cpp
 * This source file contains the definition of the described types in the IDL file.
 *
 * This file was generated by the tool gen.
 */

#ifdef _WIN32
// Remove linker warning LNK4221 on Visual Studio
namespace {
char dummy;
}  // namespace
#endif  // _WIN32

#include "metadata.h"
#include <fastcdr/Cdr.h>

#include <fastcdr/exceptions/BadParamException.h>
using namespace eprosima::fastcdr::exception;

#include <utility>

DDSMetaData::DDSMetaData()
{
    // m_mdata com.eprosima.idl.parser.typecode.ArrayTypeCode@192d3247
    memset(&m_mdata, 0, (1024*256) * 1);

}

DDSMetaData::~DDSMetaData()
{
}

DDSMetaData::DDSMetaData(
        const DDSMetaData& x)
{
    m_mdata = x.m_mdata;
}

DDSMetaData::DDSMetaData(
        DDSMetaData&& x)
{
    m_mdata = std::move(x.m_mdata);
}

DDSMetaData& DDSMetaData::operator =(
        const DDSMetaData& x)
{

    m_mdata = x.m_mdata;

    return *this;
}

DDSMetaData& DDSMetaData::operator =(
        DDSMetaData&& x)
{

    m_mdata = std::move(x.m_mdata);

    return *this;
}

bool DDSMetaData::operator ==(
        const DDSMetaData& x) const
{

    return (m_mdata == x.m_mdata);
}

bool DDSMetaData::operator !=(
        const DDSMetaData& x) const
{
    return !(*this == x);
}

size_t DDSMetaData::getMaxCdrSerializedSize(
        size_t current_alignment)
{
    size_t initial_alignment = current_alignment;


    current_alignment += ((1024*256) * 1) + eprosima::fastcdr::Cdr::alignment(current_alignment, 1);


    return current_alignment - initial_alignment;
}

size_t DDSMetaData::getCdrSerializedSize(
        const DDSMetaData& data,
        size_t current_alignment)
{
    (void)data;
    size_t initial_alignment = current_alignment;


    if ((1024*256) > 0)
    {
        current_alignment += ((1024*256) * 1) + eprosima::fastcdr::Cdr::alignment(current_alignment, 1);
    }

    return current_alignment - initial_alignment;
}

void DDSMetaData::serialize(
        eprosima::fastcdr::Cdr& scdr) const
{

    scdr << m_mdata;


}

void DDSMetaData::deserialize(
        eprosima::fastcdr::Cdr& dcdr)
{

    dcdr >> m_mdata;

}

/*!
 * @brief This function copies the value in member mdata
 * @param _mdata New value to be copied in member mdata
 */
void DDSMetaData::mdata(
        const std::array<char, 1024*256>& _mdata)
{
    m_mdata = _mdata;
}

/*!
 * @brief This function moves the value in member mdata
 * @param _mdata New value to be moved in member mdata
 */
void DDSMetaData::mdata(
        std::array<char, 1024*256>&& _mdata)
{
    m_mdata = std::move(_mdata);
}

/*!
 * @brief This function returns a constant reference to member mdata
 * @return Constant reference to member mdata
 */
const std::array<char, 1024*256>& DDSMetaData::mdata() const
{
    return m_mdata;
}

/*!
 * @brief This function returns a reference to member mdata
 * @return Reference to member mdata
 */
std::array<char, 1024*256>& DDSMetaData::mdata()
{
    return m_mdata;
}

size_t DDSMetaData::getKeyMaxCdrSerializedSize(
        size_t current_alignment)
{
    size_t current_align = current_alignment;



    return current_align;
}

bool DDSMetaData::isKeyDefined()
{
    return false;
}

void DDSMetaData::serializeKey(
        eprosima::fastcdr::Cdr& scdr) const
{
    (void) scdr;
     
}