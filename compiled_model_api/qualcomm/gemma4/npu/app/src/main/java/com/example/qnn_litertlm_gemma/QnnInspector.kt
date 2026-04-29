package com.example.qnn_litertlm_gemma

import com.qualcomm.qti.QnnDelegate
import android.util.Log

object QnnInspector {
    fun inspect() {
        Log.d("QnnInspector", "--- QnnDelegate Methods ---")
        QnnDelegate::class.java.methods.forEach {
            Log.d("QnnInspector", "Method: ${it.name}(${it.parameterTypes.joinToString { p -> p.simpleName }}) -> ${it.returnType.simpleName}")
        }
        QnnDelegate::class.java.constructors.forEach {
            Log.d("QnnInspector", "Constructor: ${it.parameterTypes.joinToString { p -> p.simpleName }}")
        }
        
        Log.d("QnnInspector", "--- QnnDelegate.Options Methods ---")
        QnnDelegate.Options::class.java.methods.forEach {
            Log.d("QnnInspector", "Method: ${it.name}(${it.parameterTypes.joinToString { p -> p.simpleName }}) -> ${it.returnType.simpleName}")
        }
    }
}
