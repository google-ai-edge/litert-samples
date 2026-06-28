
package com.google.ai.edge.examples.face_landmark

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.media.ExifInterface
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import java.util.concurrent.Executors
import kotlin.math.min

/**
 * RTMPose-m face alignment demo (WFLW, 98 landmarks), fully on the CompiledModel GPU. Center-crops a square
 * face box, estimates 98 dense landmarks, and draws the face mesh. Works on a bundled image and any picked
 * image.
 */
class MainActivity : Activity() {

    private val tag = "RTMFACE"
    private val bg = Executors.newSingleThreadExecutor()
    private var net: RtmFaceEstimator? = null
    private val pickReq = 100

    private lateinit var status: TextView
    private lateinit var meshView: MeshView

    // WFLW 98-landmark groups: (indices, closedLoop)
    private val groups = arrayOf(
        (0..32).toList() to false,    // face contour / jaw
        (33..41).toList() to false,   // left eyebrow
        (42..50).toList() to false,   // right eyebrow
        (51..54).toList() to false,   // nose bridge
        (55..59).toList() to false,   // nose bottom
        (60..67).toList() to true,    // left eye
        (68..75).toList() to true,    // right eye
        (76..87).toList() to true,    // outer lips
        (88..95).toList() to true,    // inner lips
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(24, 80, 24, 24) }
        status = TextView(this).apply { textSize = 15f; text = "Loading RTMPose-Face on GPU…" }
        val pick = Button(this).apply {
            text = "🖼  Pick image"; isEnabled = false
            setOnClickListener {
                startActivityForResult(Intent(Intent.ACTION_GET_CONTENT).apply { type = "image/*" }, pickReq)
            }
        }
        meshView = MeshView(this)
        root.addView(status); root.addView(pick)
        root.addView(meshView, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 960))
        setContentView(root)

        bg.execute {
            try {
                net = RtmFaceEstimator(this)
                try {
                    run(cropSquare(BitmapFactory.decodeStream(assets.open("test_image.jpg"))), warm = true)
                } catch (_: java.io.IOException) {
                    runOnUiThread { status.text = "Ready — pick a face image." }
                }
                runOnUiThread { pick.isEnabled = true }
            } catch (e: Throwable) {
                Log.e(tag, "load failed", e)
                runOnUiThread { status.setBackgroundColor(Color.rgb(0xFF, 0xCD, 0xD2)); status.text = "FAIL: ${e.message}" }
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != pickReq || resultCode != RESULT_OK) return
        val uri = data?.data ?: return
        runOnUiThread { status.text = "Estimating landmarks…" }
        bg.execute {
            try { run(cropSquare(loadOriented(uri)), warm = false) }
            catch (e: Throwable) { Log.e(tag, "estimate failed", e); runOnUiThread { status.text = "Failed: ${e.message}" } }
        }
    }

    private fun run(crop: Bitmap, warm: Boolean) {
        val n = net!!
        val rgb = bitmapToRgb(crop)
        if (warm) n.estimate(rgb)
        val t0 = System.nanoTime()
        val pts = n.estimate(rgb)
        val ms = (System.nanoTime() - t0) / 1_000_000
        Log.i(tag, "estimate ${ms}ms pts=${pts.size}")
        runOnUiThread {
            status.setBackgroundColor(Color.rgb(0xC8, 0xE6, 0xC9))
            status.text = "On-device GPU face mesh ✓  ${ms} ms  ·  98 landmarks  ·  RTMPose-m WFLW, CompiledModel GPU"
            meshView.set(crop, pts, groups); meshView.invalidate()
        }
    }

    private fun loadOriented(uri: Uri): Bitmap {
        val bm = contentResolver.openInputStream(uri).use { BitmapFactory.decodeStream(it) } ?: error("cannot decode image")
        val rot = contentResolver.openInputStream(uri).use {
            when (ExifInterface(it!!).getAttributeInt(ExifInterface.TAG_ORIENTATION, 1)) {
                ExifInterface.ORIENTATION_ROTATE_90 -> 90f
                ExifInterface.ORIENTATION_ROTATE_180 -> 180f
                ExifInterface.ORIENTATION_ROTATE_270 -> 270f
                else -> 0f
            }
        }
        if (rot == 0f) return bm
        return Bitmap.createBitmap(bm, 0, 0, bm.width, bm.height, Matrix().apply { postRotate(rot) }, true)
    }

    private fun cropSquare(src: Bitmap): Bitmap {
        val s = min(src.width, src.height)
        val crop = Bitmap.createBitmap(src, (src.width - s) / 2, (src.height - s) / 2, s, s)
        return Bitmap.createScaledBitmap(crop, RtmFaceEstimator.W, RtmFaceEstimator.H, true)
    }

    private fun bitmapToRgb(bm: Bitmap): FloatArray {
        val n = bm.width * bm.height; val px = IntArray(n)
        bm.getPixels(px, 0, bm.width, 0, 0, bm.width, bm.height)
        val out = FloatArray(n * 3)
        for (i in 0 until n) {
            val p = px[i]
            out[i * 3] = ((p shr 16) and 0xFF).toFloat(); out[i * 3 + 1] = ((p shr 8) and 0xFF).toFloat()
            out[i * 3 + 2] = (p and 0xFF).toFloat()
        }
        return out
    }

    override fun onDestroy() { super.onDestroy(); bg.shutdown(); net?.close() }

    class MeshView(ctx: Context) : View(ctx) {
        private var bm: Bitmap? = null
        private var pts: List<RtmFaceEstimator.Point> = emptyList()
        private var groups: Array<Pair<List<Int>, Boolean>> = emptyArray()
        private val line = Paint().apply { color = Color.rgb(0, 230, 0); strokeWidth = 2.5f; style = Paint.Style.STROKE; isAntiAlias = true }
        private val dot = Paint().apply { color = Color.rgb(255, 40, 40); isAntiAlias = true }
        private val imgPaint = Paint().apply { isFilterBitmap = true }

        fun set(b: Bitmap, p: List<RtmFaceEstimator.Point>, g: Array<Pair<List<Int>, Boolean>>) { bm = b; pts = p; groups = g }

        override fun onDraw(canvas: Canvas) {
            val b = bm ?: return
            val s = min(width.toFloat() / b.width, height.toFloat() / b.height)
            val w = b.width * s; val h = b.height * s
            val ox = (width - w) / 2; val oy = (height - h) / 2
            canvas.drawBitmap(b, null, android.graphics.RectF(ox, oy, ox + w, oy + h), imgPaint)
            fun px(i: Int) = ox + pts[i].x * s
            fun py(i: Int) = oy + pts[i].y * s
            for ((ids, closed) in groups) {
                for (j in 0 until ids.size - 1) canvas.drawLine(px(ids[j]), py(ids[j]), px(ids[j + 1]), py(ids[j + 1]), line)
                if (closed && ids.size > 1) canvas.drawLine(px(ids.last()), py(ids.last()), px(ids.first()), py(ids.first()), line)
            }
            for (i in pts.indices) canvas.drawCircle(px(i), py(i), 2.5f, dot)
        }
    }
}
