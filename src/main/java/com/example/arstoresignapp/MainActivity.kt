package com.example.arstoresignapp

import android.graphics.Bitmap
import android.graphics.Canvas
import android.os.Bundle
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.google.ar.core.Frame
import com.google.ar.sceneform.ux.ArFragment
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {
    private lateinit var arFragment: ArFragment
    private lateinit var infoTextView: TextView
    private lateinit var tflite: Interpreter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // AR Fragment 초기화
        try {
            arFragment = supportFragmentManager.findFragmentById(R.id.arFragment) as ArFragment
            infoTextView = findViewById(R.id.infoTextView)
        } catch (e: Exception) {
            Toast.makeText(this, "ARCore를 초기화할 수 없습니다.", Toast.LENGTH_SHORT).show()
            finish()
            return
        }

        // TensorFlow Lite 모델 로드
        val model = loadModelFile()
        tflite = Interpreter(model)

        // AR Scene 업데이트 리스너 등록
        arFragment.arSceneView.scene.addOnUpdateListener { frameTime ->
            arFragment.onUpdate(frameTime)
            onUpdate()
        }
    }

    private fun loadModelFile(): ByteBuffer {
        val fileName = "alexnet_model.tflite"
        val assetFileDescriptor = assets.openFd(fileName)
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun onUpdate() {
        val frame: Frame = arFragment.arSceneView.arFrame ?: return

        // 이미지 캡처 및 전처리
        val bitmap = captureBitmap()
        val preprocessedImage = preprocessImage(bitmap)

        // TensorFlow Lite 추론 수행
        val prediction = performInference(preprocessedImage)

        // 결과 처리 및 화면에 정보 표시
        displayPrediction(prediction)
    }

    private fun captureBitmap(): Bitmap {
        val arSceneView = arFragment.arSceneView

        // ARSceneView의 크기로 비트맵 생성
        val bitmap = Bitmap.createBitmap(
            arSceneView.width,
            arSceneView.height,
            Bitmap.Config.ARGB_8888
        )

        val canvas = Canvas(bitmap)
        arSceneView.draw(canvas)
        return bitmap
    }

    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val inputBuffer = ByteBuffer.allocateDirect(4 * 227 * 227 * 3)
        inputBuffer.order(ByteOrder.nativeOrder())

        // 이미지 크기 조정 및 정규화
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 227, 227, true)
        val intValues = IntArray(227 * 227)
        resizedBitmap.getPixels(
            intValues,
            0,
            resizedBitmap.width,
            0,
            0,
            resizedBitmap.width,
            resizedBitmap.height
        )

        var pixelIndex = 0
        for (y in 0 until 227) {
            for (x in 0 until 227) {
                val pixelValue = intValues[pixelIndex++]

                // ARGB 값을 RGB로 변환하고 정규화
                inputBuffer.putFloat(((pixelValue shr 16 and 0xFF) / 255.0f))
                inputBuffer.putFloat(((pixelValue shr 8 and 0xFF) / 255.0f))  // Green
                inputBuffer.putFloat(((pixelValue and 0xFF) / 255.0f))        // Blue
            }
        }
        return inputBuffer
    }

    private fun performInference(inputBuffer: ByteBuffer): String {
        // 추론 결과 저장을 위한 배열
        val outputBuffer = ByteBuffer.allocateDirect(4 * 2).order(ByteOrder.nativeOrder())
        tflite.run(inputBuffer, outputBuffer)

        // 예측 결과 해석
        outputBuffer.rewind()
        val probabilities = FloatArray(2)
        outputBuffer.asFloatBuffer().get(probabilities)

        // 간판의 레이블 추출
        return if (probabilities[0] > probabilities[1]) "간판1" else "간판2"
    }

    private fun displayPrediction(prediction: String) {
        // 예측된 간판 정보 표시
        infoTextView.text = "인식된 간판: $prediction"
    }
}
