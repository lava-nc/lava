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
 * @file DDSMetaData.h
 * This header file contains the declaration of the described types in the IDL file.
 *
 * This file was generated by the tool gen.
 */

#ifndef _FAST_DDS_GENERATED_DDSMETADATA_MSG_DDSMETADATA_H_
#define _FAST_DDS_GENERATED_DDSMETADATA_MSG_DDSMETADATA_H_


#include <fastrtps/utils/fixed_size_string.hpp>

#include <stdint.h>
#include <array>
#include <string>
#include <vector>
#include <map>
#include <bitset>

#if defined(_WIN32)
#if defined(EPROSIMA_USER_DLL_EXPORT)
#define eProsima_user_DllExport __declspec( dllexport )
#else
#define eProsima_user_DllExport
#endif  // EPROSIMA_USER_DLL_EXPORT
#else
#define eProsima_user_DllExport
#endif  // _WIN32

#if defined(_WIN32)
#if defined(EPROSIMA_USER_DLL_EXPORT)
#if defined(DDSMetaData_SOURCE)
#define DDSMetaData_DllAPI __declspec( dllexport )
#else
#define DDSMetaData_DllAPI __declspec( dllimport )
#endif // DDSMetaData_SOURCE
#else
#define DDSMetaData_DllAPI
#endif  // EPROSIMA_USER_DLL_EXPORT
#else
#define DDSMetaData_DllAPI
#endif // _WIN32

namespace eprosima {
namespace fastcdr {
class Cdr;
} // namespace fastcdr
} // namespace eprosima


namespace ddsmetadata {
    namespace msg {
        typedef std::array<int64_t, 5> int64__5;
        /*!
         * @brief This class represents the structure DDSMetaData defined by the user in the IDL file.
         * @ingroup DDSMETADATA
         */
        class DDSMetaData
        {
        public:

            /*!
             * @brief Default constructor.
             */
            eProsima_user_DllExport DDSMetaData();

            /*!
             * @brief Default destructor.
             */
            eProsima_user_DllExport ~DDSMetaData();

            /*!
             * @brief Copy constructor.
             * @param x Reference to the object ddsmetadata::msg::DDSMetaData that will be copied.
             */
            eProsima_user_DllExport DDSMetaData(
                    const DDSMetaData& x);

            /*!
             * @brief Move constructor.
             * @param x Reference to the object ddsmetadata::msg::DDSMetaData that will be copied.
             */
            eProsima_user_DllExport DDSMetaData(
                    DDSMetaData&& x);

            /*!
             * @brief Copy assignment.
             * @param x Reference to the object ddsmetadata::msg::DDSMetaData that will be copied.
             */
            eProsima_user_DllExport DDSMetaData& operator =(
                    const DDSMetaData& x);

            /*!
             * @brief Move assignment.
             * @param x Reference to the object ddsmetadata::msg::DDSMetaData that will be copied.
             */
            eProsima_user_DllExport DDSMetaData& operator =(
                    DDSMetaData&& x);

            /*!
             * @brief Comparison operator.
             * @param x ddsmetadata::msg::DDSMetaData object to compare.
             */
            eProsima_user_DllExport bool operator ==(
                    const DDSMetaData& x) const;

            /*!
             * @brief Comparison operator.
             * @param x ddsmetadata::msg::DDSMetaData object to compare.
             */
            eProsima_user_DllExport bool operator !=(
                    const DDSMetaData& x) const;

            /*!
             * @brief This function sets a value in member nd
             * @param _nd New value for member nd
             */
            eProsima_user_DllExport void nd(
                    int64_t _nd);

            /*!
             * @brief This function returns the value of member nd
             * @return Value of member nd
             */
            eProsima_user_DllExport int64_t nd() const;

            /*!
             * @brief This function returns a reference to member nd
             * @return Reference to member nd
             */
            eProsima_user_DllExport int64_t& nd();

            /*!
             * @brief This function sets a value in member type
             * @param _type New value for member type
             */
            eProsima_user_DllExport void type(
                    int64_t _type);

            /*!
             * @brief This function returns the value of member type
             * @return Value of member type
             */
            eProsima_user_DllExport int64_t type() const;

            /*!
             * @brief This function returns a reference to member type
             * @return Reference to member type
             */
            eProsima_user_DllExport int64_t& type();

            /*!
             * @brief This function sets a value in member elsize
             * @param _elsize New value for member elsize
             */
            eProsima_user_DllExport void elsize(
                    int64_t _elsize);

            /*!
             * @brief This function returns the value of member elsize
             * @return Value of member elsize
             */
            eProsima_user_DllExport int64_t elsize() const;

            /*!
             * @brief This function returns a reference to member elsize
             * @return Reference to member elsize
             */
            eProsima_user_DllExport int64_t& elsize();

            /*!
             * @brief This function sets a value in member total_size
             * @param _total_size New value for member total_size
             */
            eProsima_user_DllExport void total_size(
                    int64_t _total_size);

            /*!
             * @brief This function returns the value of member total_size
             * @return Value of member total_size
             */
            eProsima_user_DllExport int64_t total_size() const;

            /*!
             * @brief This function returns a reference to member total_size
             * @return Reference to member total_size
             */
            eProsima_user_DllExport int64_t& total_size();

            /*!
             * @brief This function copies the value in member dims
             * @param _dims New value to be copied in member dims
             */
            eProsima_user_DllExport void dims(
                    const ddsmetadata::msg::int64__5& _dims);

            /*!
             * @brief This function moves the value in member dims
             * @param _dims New value to be moved in member dims
             */
            eProsima_user_DllExport void dims(
                    ddsmetadata::msg::int64__5&& _dims);

            /*!
             * @brief This function returns a constant reference to member dims
             * @return Constant reference to member dims
             */
            eProsima_user_DllExport const ddsmetadata::msg::int64__5& dims() const;

            /*!
             * @brief This function returns a reference to member dims
             * @return Reference to member dims
             */
            eProsima_user_DllExport ddsmetadata::msg::int64__5& dims();
            /*!
             * @brief This function copies the value in member strides
             * @param _strides New value to be copied in member strides
             */
            eProsima_user_DllExport void strides(
                    const ddsmetadata::msg::int64__5& _strides);

            /*!
             * @brief This function moves the value in member strides
             * @param _strides New value to be moved in member strides
             */
            eProsima_user_DllExport void strides(
                    ddsmetadata::msg::int64__5&& _strides);

            /*!
             * @brief This function returns a constant reference to member strides
             * @return Constant reference to member strides
             */
            eProsima_user_DllExport const ddsmetadata::msg::int64__5& strides() const;

            /*!
             * @brief This function returns a reference to member strides
             * @return Reference to member strides
             */
            eProsima_user_DllExport ddsmetadata::msg::int64__5& strides();
            /*!
             * @brief This function copies the value in member mdata
             * @param _mdata New value to be copied in member mdata
             */
            eProsima_user_DllExport void mdata(
                    const std::vector<uint8_t>& _mdata);

            /*!
             * @brief This function moves the value in member mdata
             * @param _mdata New value to be moved in member mdata
             */
            eProsima_user_DllExport void mdata(
                    std::vector<uint8_t>&& _mdata);

            /*!
             * @brief This function returns a constant reference to member mdata
             * @return Constant reference to member mdata
             */
            eProsima_user_DllExport const std::vector<uint8_t>& mdata() const;

            /*!
             * @brief This function returns a reference to member mdata
             * @return Reference to member mdata
             */
            eProsima_user_DllExport std::vector<uint8_t>& mdata();

            /*!
             * @brief This function returns the maximum serialized size of an object
             * depending on the buffer alignment.
             * @param current_alignment Buffer alignment.
             * @return Maximum serialized size.
             */
            eProsima_user_DllExport static size_t getMaxCdrSerializedSize(
                    size_t current_alignment = 0);

            /*!
             * @brief This function returns the serialized size of a data depending on the buffer alignment.
             * @param data Data which is calculated its serialized size.
             * @param current_alignment Buffer alignment.
             * @return Serialized size.
             */
            eProsima_user_DllExport static size_t getCdrSerializedSize(
                    const ddsmetadata::msg::DDSMetaData& data,
                    size_t current_alignment = 0);


            /*!
             * @brief This function serializes an object using CDR serialization.
             * @param cdr CDR serialization object.
             */
            eProsima_user_DllExport void serialize(
                    eprosima::fastcdr::Cdr& cdr) const;

            /*!
             * @brief This function deserializes an object using CDR serialization.
             * @param cdr CDR serialization object.
             */
            eProsima_user_DllExport void deserialize(
                    eprosima::fastcdr::Cdr& cdr);



            /*!
             * @brief This function returns the maximum serialized size of the Key of an object
             * depending on the buffer alignment.
             * @param current_alignment Buffer alignment.
             * @return Maximum serialized size.
             */
            eProsima_user_DllExport static size_t getKeyMaxCdrSerializedSize(
                    size_t current_alignment = 0);

            /*!
             * @brief This function tells you if the Key has been defined for this type
             */
            eProsima_user_DllExport static bool isKeyDefined();

            /*!
             * @brief This function serializes the key members of an object using CDR serialization.
             * @param cdr CDR serialization object.
             */
            eProsima_user_DllExport void serializeKey(
                    eprosima::fastcdr::Cdr& cdr) const;

        private:

            int64_t m_nd;
            int64_t m_type;
            int64_t m_elsize;
            int64_t m_total_size;
            ddsmetadata::msg::int64__5 m_dims;
            ddsmetadata::msg::int64__5 m_strides;
            std::vector<uint8_t> m_mdata;
        };
    } // namespace msg
} // namespace ddsmetadata

#endif // _FAST_DDS_GENERATED_DDSMETADATA_MSG_DDSMETADATA_H_