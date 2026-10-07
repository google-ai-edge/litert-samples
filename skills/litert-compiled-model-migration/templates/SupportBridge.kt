// TODO: replace with the app's package.
package com.example.validation

import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.LiteRtException
import com.google.ai.edge.litert.TensorBuffer
import com.google.ai.edge.litert.TensorType
import com.google.ai.edge.litert.support.image.TensorImage
import com.google.ai.edge.litert.support.tensorbuffer.TensorBuffer as SupportTensorBuffer
import java.nio.ByteBuffer
import org.tensorflow.lite.DataType

/**
 * Moves data between LiteRT Support tensors (litert-support 2.3.0: [TensorImage] and its
 * [SupportTensorBuffer]) and the buffers of a [CompiledModel]. Copy this file only into an app
 * that keeps LiteRT Support (LiteRtModel.kt does not need it). Both libraries have a class named
 * TensorBuffer, so this file imports the Support one as SupportTensorBuffer.
 *
 * Create one per model after CompiledModel.create, with the names of the input and the output
 * tensor (`args_0` / `output_0` for LiteRT-Torch exports, the graph's tensor names otherwise,
 * which Step 0 prints). It reads the byte size of both tensors once and refuses a Support tensor
 * of another size, as the Interpreter did for a ByteBuffer. The Kotlin buffer API does not check:
 * writeFloat of a shorter array leaves the rest of the input unchanged, and writeInt8 of a uint8
 * image into a float input runs and returns NaN. For a float input it also refuses a uint8 image
 * (a TensorImage that did not go through NormalizeOp). Call [write], run and [read] on the
 * model's dispatcher, and close the buffers after the last [read] has returned. A closed buffer
 * throws IllegalStateException.
 */
class SupportBridge(model: CompiledModel, inputName: String, outputName: String) {
    private val inputBytes: Int = model.getInputBufferRequirements(inputName).bufferSize
    private val outputBytes: Int = model.getOutputBufferRequirements(outputName).bufferSize
    // Null for a tensor type the Kotlin API does not name (uint8 in 2.3.0).
    private val inputElementType: TensorType.ElementType? =
        try {
            model.getInputTensorType(inputName).elementType
        } catch (e: LiteRtException) {
            null
        }

    /** Writes a processed image into the input: FLOAT32 via writeFloat, UINT8 via writeInt8. */
    fun write(input: TensorBuffer, image: TensorImage) = write(input, image.tensorBuffer)

    fun write(input: TensorBuffer, tensor: SupportTensorBuffer) {
        val bytes = tensor.flatSize * tensor.typeSize
        require(bytes == inputBytes) {
            "Support tensor of $bytes bytes (${tensor.flatSize} ${tensor.dataType} values) " +
                "for a model input of $inputBytes bytes"
        }
        val floatInput = inputElementType == TensorType.ElementType.FLOAT
        require(!floatInput || tensor.dataType == DataType.FLOAT32) {
            "${tensor.dataType} Support tensor for a FLOAT input: " +
                "the image did not go through NormalizeOp"
        }
        when (tensor.dataType) {
            DataType.FLOAT32 -> input.writeFloat(tensor.floatArray)
            DataType.UINT8 -> {
                val pixels = ByteArray(tensor.flatSize)
                tensor.buffer.duplicate().apply { rewind() }.get(pixels)
                input.writeInt8(pixels)
            }
            else -> throw IllegalArgumentException(
                "SupportBridge handles FLOAT32 and UINT8; got ${tensor.dataType}"
            )
        }
    }

    /** Reads the output into a Support tensor of [shape] and [dataType], for TensorLabel. */
    fun read(output: TensorBuffer, shape: IntArray, dataType: DataType): SupportTensorBuffer {
        val tensor = SupportTensorBuffer.createFixedSize(shape, dataType)
        val bytes = tensor.flatSize * tensor.typeSize
        require(bytes == outputBytes) {
            "Support tensor of $bytes bytes (${shape.toList()} $dataType) " +
                "for a model output of $outputBytes bytes"
        }
        when (dataType) {
            DataType.FLOAT32 -> tensor.loadArray(output.readFloat())
            DataType.UINT8 -> tensor.loadBuffer(ByteBuffer.wrap(output.readInt8()))
            else -> throw IllegalArgumentException(
                "SupportBridge handles FLOAT32 and UINT8; got $dataType"
            )
        }
        return tensor
    }
}
