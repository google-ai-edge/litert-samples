package com.example.fastvlm.adapter

import android.text.format.DateFormat
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import coil.load
import coil.transform.RoundedCornersTransformation
import com.example.fastvlm.R
import com.example.fastvlm.model.ChatMessage
import com.example.fastvlm.model.Role
import java.util.Date

class ChatAdapter : ListAdapter<ChatMessage, RecyclerView.ViewHolder>(ChatDiffCallback()) {

    companion object {
        private const val VIEW_TYPE_USER = 0
        private const val VIEW_TYPE_ASSISTANT = 1
    }

    override fun getItemViewType(position: Int): Int {
        return when (getItem(position).role) {
            Role.USER -> VIEW_TYPE_USER
            Role.ASSISTANT -> VIEW_TYPE_ASSISTANT
            else -> VIEW_TYPE_ASSISTANT
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        val inflater = LayoutInflater.from(parent.context)
        return when (viewType) {
            VIEW_TYPE_USER -> {
                val view = inflater.inflate(R.layout.item_message_user, parent, false)
                UserMessageViewHolder(view)
            }
            else -> {
                val view = inflater.inflate(R.layout.item_message_assistant, parent, false)
                AssistantMessageViewHolder(view)
            }
        }
    }

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        val message = getItem(position)
        when (holder) {
            is UserMessageViewHolder -> holder.bind(message)
            is AssistantMessageViewHolder -> holder.bind(message)
        }
    }

    // ─── User Message ViewHolder ───────────────────────────────

    class UserMessageViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val tvMessage: TextView = itemView.findViewById(R.id.tvUserMessage)
        private val tvTimestamp: TextView = itemView.findViewById(R.id.tvUserTimestamp)
        private val ivImage: ImageView = itemView.findViewById(R.id.ivUserImage)

        fun bind(message: ChatMessage) {
            tvMessage.text = message.text

            // Format timestamp
            val timeFormat = DateFormat.getTimeFormat(itemView.context)
            tvTimestamp.text = timeFormat.format(Date(message.timestamp))

            // Show image if present
            if (message.imageUri != null) {
                ivImage.visibility = View.VISIBLE
                ivImage.load(message.imageUri) {
                    crossfade(true)
                    transformations(RoundedCornersTransformation(24f))
                    placeholder(R.drawable.bg_image_rounded)
                }
            } else {
                ivImage.visibility = View.GONE
            }
        }
    }

    // ─── Assistant Message ViewHolder ──────────────────────────

    class AssistantMessageViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val tvMessage: TextView = itemView.findViewById(R.id.tvAssistantMessage)
        private val tvTypingIndicator: TextView = itemView.findViewById(R.id.tvTypingIndicator)

        fun bind(message: ChatMessage) {
            if (message.text.isEmpty()) {
                // Show typing indicator while generating
                tvMessage.visibility = View.GONE
                tvTypingIndicator.visibility = View.VISIBLE
            } else {
                tvMessage.visibility = View.VISIBLE
                tvMessage.text = message.text
                tvTypingIndicator.visibility = View.GONE
            }
        }
    }

    // ─── DiffUtil ──────────────────────────────────────────────

    class ChatDiffCallback : DiffUtil.ItemCallback<ChatMessage>() {
        override fun areItemsTheSame(oldItem: ChatMessage, newItem: ChatMessage): Boolean {
            return oldItem.id == newItem.id
        }

        override fun areContentsTheSame(oldItem: ChatMessage, newItem: ChatMessage): Boolean {
            return oldItem == newItem
        }
    }
}
