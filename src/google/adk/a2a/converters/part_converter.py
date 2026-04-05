# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
module containing utilities for conversion between A2A Part and Google GenAI Part
"""

from __future__ import annotations

import base64
from collections.abc import Callable
import json
import logging
from typing import List
from typing import Optional
from typing import Union

from a2a import types as a2a_types
from google.genai import types as genai_types
from google.protobuf import struct_pb2
from google.protobuf.json_format import MessageToDict, MessageToJson

from ..experimental import a2a_experimental
from .utils import _get_adk_metadata_key


logger = logging.getLogger('google_adk.' + __name__)

A2A_DATA_PART_METADATA_TYPE_KEY = 'type'
A2A_DATA_PART_METADATA_IS_LONG_RUNNING_KEY = 'is_long_running'
A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL = 'function_call'
A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE = 'function_response'
A2A_DATA_PART_METADATA_TYPE_CODE_EXECUTION_RESULT = 'code_execution_result'
A2A_DATA_PART_METADATA_TYPE_EXECUTABLE_CODE = 'executable_code'
A2A_DATA_PART_TEXT_MIME_TYPE = 'text/plain'
A2A_DATA_PART_START_TAG = b'<a2a_datapart_json>'
A2A_DATA_PART_END_TAG = b'</a2a_datapart_json>'


A2APartToGenAIPartConverter = Callable[
    [a2a_types.Part], Union[Optional[genai_types.Part], List[genai_types.Part]]
]
GenAIPartToA2APartConverter = Callable[
    [genai_types.Part],
    Union[Optional[a2a_types.Part], List[a2a_types.Part]],
]


def _has_field(part: a2a_types.Part, field_name: str) -> bool:
  """Returns whether a proto-like part has a populated field."""
  has_field = getattr(part, 'HasField', None)
  if not callable(has_field):
    return False
  try:
    result = has_field(field_name)
  except Exception:
    return False
  return isinstance(result, bool) and result


def _get_metadata_value(part: a2a_types.Part, key: str):
  """Returns a metadata value from a proto Struct or dict-like metadata."""
  metadata = getattr(part, 'metadata', None)
  if not metadata:
    return None
  try:
    return metadata.get(key)
  except AttributeError:
    try:
      return metadata[key]
    except Exception:
      return None


@a2a_experimental
def convert_a2a_part_to_genai_part(
    a2a_part: a2a_types.Part,
) -> Optional[genai_types.Part]:
  """Convert an A2A Part to a Google GenAI Part."""
  
  if _has_field(a2a_part, 'text'):
    thought = None
    thought = _get_metadata_value(a2a_part, _get_adk_metadata_key('thought'))
    return genai_types.Part(text=a2a_part.text, thought=thought)

  if _has_field(a2a_part, 'url'):
    return genai_types.Part(
        file_data=genai_types.FileData(
            file_uri=a2a_part.url,
            mime_type=a2a_part.media_type,
            display_name=a2a_part.filename,
        )
    )

  if _has_field(a2a_part, 'raw'):
    return genai_types.Part(
        inline_data=genai_types.Blob(
            data=a2a_part.raw,
            mime_type=a2a_part.media_type,
            display_name=a2a_part.filename,
        )
    )

  if _has_field(a2a_part, 'data'):
    # Convert the Data Part to funcall and function response.
    # This is mainly for converting human in the loop and auth request and
    # response.
    # TODO once A2A defined how to service such information, migrate below
    # logic accordingly
    part_type = _get_metadata_value(
        a2a_part, _get_adk_metadata_key(A2A_DATA_PART_METADATA_TYPE_KEY)
    )
    if part_type is not None:
      try:
          data_dict = MessageToDict(a2a_part.data)
      except Exception:
          data_dict = {}

      if (
          part_type
          == A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL
      ):
        # Restore thought_signature if present
        thought_signature = None
        thought_sig_key = _get_adk_metadata_key('thought_signature')
        sig_value = _get_metadata_value(a2a_part, thought_sig_key)
        if sig_value is not None:
          if isinstance(sig_value, bytes):
            thought_signature = sig_value
          elif isinstance(sig_value, str):
            try:
              thought_signature = base64.b64decode(sig_value)
            except Exception:
              logger.warning(
                  'Failed to decode thought_signature: %s', sig_value
              )
        return genai_types.Part(
            function_call=genai_types.FunctionCall.model_validate(
                data_dict, by_alias=True
            ),
            thought_signature=thought_signature,
        )
      if (
          part_type
          == A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE
      ):
        return genai_types.Part(
            function_response=genai_types.FunctionResponse.model_validate(
                data_dict, by_alias=True
            )
        )
      if (
          part_type
          == A2A_DATA_PART_METADATA_TYPE_CODE_EXECUTION_RESULT
      ):
        return genai_types.Part(
            code_execution_result=genai_types.CodeExecutionResult.model_validate(
                data_dict, by_alias=True
            )
        )
      if (
          part_type
          == A2A_DATA_PART_METADATA_TYPE_EXECUTABLE_CODE
      ):
        return genai_types.Part(
            executable_code=genai_types.ExecutableCode.model_validate(
                data_dict, by_alias=True
            )
        )
    
    # Extract the JSON payload using MessageToJson 
    # and then encode to bytes for inline_data
    try:
      data_json = MessageToJson(a2a_part.data, preserving_proto_field_name=True)
    except Exception as e:
      logger.warning('Failed to render data to json: %s', e)
      data_json = "{}"
    
    return genai_types.Part(
        inline_data=genai_types.Blob(
            data=A2A_DATA_PART_START_TAG
            + data_json.encode('utf-8')
            + A2A_DATA_PART_END_TAG,
            mime_type=A2A_DATA_PART_TEXT_MIME_TYPE,
        )
    )

  logger.warning(
      'Cannot convert unsupported A2A part: %s',
      a2a_part,
  )
  return None


@a2a_experimental
def convert_genai_part_to_a2a_part(
    part: genai_types.Part,
) -> Optional[a2a_types.Part]:
  """Convert a Google GenAI Part to an A2A Part."""
  if part is None:
    logger.warning('Cannot convert unsupported GenAI part: %s', part)
    return None

  if part.text:
    a2a_part = a2a_types.Part(text=part.text)
    if part.thought is not None:
      # Struct initialization of metadata
      a2a_part.metadata.update({_get_adk_metadata_key('thought'): part.thought})
    return a2a_part

  if part.file_data:
    return a2a_types.Part(
        url=part.file_data.file_uri,
        media_type=part.file_data.mime_type,
        filename=part.file_data.display_name,
    )

  if part.inline_data:
    if (
        part.inline_data.mime_type == A2A_DATA_PART_TEXT_MIME_TYPE
        and part.inline_data.data is not None
        and part.inline_data.data.startswith(A2A_DATA_PART_START_TAG)
        and part.inline_data.data.endswith(A2A_DATA_PART_END_TAG)
    ):
      extracted_json = part.inline_data.data[
          len(A2A_DATA_PART_START_TAG) : -len(A2A_DATA_PART_END_TAG)
      ]
      try:
          data_dict = json.loads(extracted_json)
          v = struct_pb2.Value()
          from google.protobuf.json_format import ParseDict
          ParseDict(data_dict, v)
          return a2a_types.Part(data=v)
      except Exception as e:
          logger.warning('Failed to parse GenAI datapart json: %s', e)
          return a2a_types.Part(data=struct_pb2.Value())

    # The default case for inline_data is to convert it to raw Part.
    a2a_part = a2a_types.Part(
        raw=part.inline_data.data,
        media_type=part.inline_data.mime_type,
        filename=part.inline_data.display_name,
    )

    if part.video_metadata:
      video_dict = part.video_metadata.model_dump(by_alias=True, exclude_none=True)
      a2a_part.metadata.update(
          {_get_adk_metadata_key('video_metadata'): video_dict}
      )

    return a2a_part

  # Convert the funcall and function response to A2A Data Part.
  # This is mainly for converting human in the loop and auth request and
  # response.
  # TODO once A2A defined how to service such information, migrate below
  # logic accordingly
  if part.function_call:
    fc_metadata = {
        _get_adk_metadata_key(
            A2A_DATA_PART_METADATA_TYPE_KEY
        ): A2A_DATA_PART_METADATA_TYPE_FUNCTION_CALL
    }
    # Preserve thought_signature if present
    if part.thought_signature is not None:
      fc_metadata[_get_adk_metadata_key('thought_signature')] = (
          base64.b64encode(part.thought_signature).decode('utf-8')
      )
      
    fc_dict = part.function_call.model_dump(
        by_alias=True, exclude_none=True
    )
    from google.protobuf.json_format import ParseDict
    v = struct_pb2.Value()
    ParseDict(fc_dict, v)
    
    a2a_part = a2a_types.Part(data=v)
    a2a_part.metadata.update(fc_metadata)
    return a2a_part

  if part.function_response:
    fr_dict = part.function_response.model_dump(
        by_alias=True, exclude_none=True
    )
    from google.protobuf.json_format import ParseDict
    v = struct_pb2.Value()
    ParseDict(fr_dict, v)
    
    a2a_part = a2a_types.Part(data=v)
    a2a_part.metadata.update({
        _get_adk_metadata_key(
            A2A_DATA_PART_METADATA_TYPE_KEY
        ): A2A_DATA_PART_METADATA_TYPE_FUNCTION_RESPONSE
    })
    return a2a_part

  if part.code_execution_result:
    cer_dict = part.code_execution_result.model_dump(
        by_alias=True, exclude_none=True
    )
    from google.protobuf.json_format import ParseDict
    v = struct_pb2.Value()
    ParseDict(cer_dict, v)
    
    a2a_part = a2a_types.Part(data=v)
    a2a_part.metadata.update({
        _get_adk_metadata_key(
            A2A_DATA_PART_METADATA_TYPE_KEY
        ): A2A_DATA_PART_METADATA_TYPE_CODE_EXECUTION_RESULT
    })
    return a2a_part

  if part.executable_code:
    ec_dict = part.executable_code.model_dump(
        by_alias=True, exclude_none=True
    )
    from google.protobuf.json_format import ParseDict
    v = struct_pb2.Value()
    ParseDict(ec_dict, v)
    
    a2a_part = a2a_types.Part(data=v)
    a2a_part.metadata.update({
        _get_adk_metadata_key(
            A2A_DATA_PART_METADATA_TYPE_KEY
        ): A2A_DATA_PART_METADATA_TYPE_EXECUTABLE_CODE
    })
    return a2a_part

  logger.warning(
      'Cannot convert unsupported part for Google GenAI part: %s',
      part,
  )
  return None
