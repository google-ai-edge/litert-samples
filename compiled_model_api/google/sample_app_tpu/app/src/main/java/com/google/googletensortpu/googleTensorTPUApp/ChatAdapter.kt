package com.google.googletensortpu.googleTensorTPUApp

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

class ChatAdapter(private val messages: List<ChatMessage>) :
    RecyclerView.Adapter<RecyclerView.ViewHolder>() {

    private val VIEW_TYPE_USER = 1
    private val VIEW_TYPE_BOT = 2

    private var mediaPlayer: android.media.MediaPlayer? = null
    private var playingPath: String? = null
    private var playingIcon: ImageView? = null

    class UserViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val messageText: TextView = view.findViewById(R.id.messageText)
        val messageImage: ImageView = view.findViewById(R.id.messageImage)
        val audioLayout: View? = view.findViewById(R.id.audioLayout)
        val audioPlayIcon: ImageView? = view.findViewById(R.id.audioPlayIcon)
    }

    class BotViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val messageText: TextView = view.findViewById(R.id.messageText)
        val messageImage: ImageView = view.findViewById(R.id.messageImage)
    }

    override fun getItemViewType(position: Int): Int {
        return if (messages[position].isUser) VIEW_TYPE_USER else VIEW_TYPE_BOT
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        return if (viewType == VIEW_TYPE_USER) {
            val view = LayoutInflater.from(parent.context)
                .inflate(R.layout.item_chat_user, parent, false)
            UserViewHolder(view)
        } else {
            val view = LayoutInflater.from(parent.context)
                .inflate(R.layout.item_chat_bot, parent, false)
            BotViewHolder(view)
        }
    }

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        val message = messages[position]
        if (holder is UserViewHolder) {
            holder.messageText.text = message.text
            if (message.image != null) {
                holder.messageImage.visibility = View.VISIBLE
                holder.messageImage.setImageBitmap(message.image)
            } else {
                holder.messageImage.visibility = View.GONE
            }

            if (message.audioPath != null) {
                holder.audioLayout?.visibility = View.VISIBLE
                
                if (playingPath == message.audioPath && mediaPlayer?.isPlaying == true) {
                    holder.audioPlayIcon?.setImageResource(android.R.drawable.ic_media_pause)
                    playingIcon = holder.audioPlayIcon
                } else {
                    holder.audioPlayIcon?.setImageResource(android.R.drawable.ic_media_play)
                }

                holder.audioLayout?.setOnClickListener {
                    holder.audioPlayIcon?.let { icon ->
                        playAudio(message.audioPath, icon)
                    }
                }
            } else {
                holder.audioLayout?.visibility = View.GONE
            }
        } else if (holder is BotViewHolder) {
            holder.messageText.text = message.text
            if (message.image != null) {
                holder.messageImage.visibility = View.VISIBLE
                holder.messageImage.setImageBitmap(message.image)
            } else {
                holder.messageImage.visibility = View.GONE
            }
        }
    }

    private fun playAudio(path: String, iconView: ImageView) {
        if (playingPath == path && mediaPlayer?.isPlaying == true) {
            mediaPlayer?.stop()
            mediaPlayer?.release()
            mediaPlayer = null
            playingPath = null
            iconView.setImageResource(android.R.drawable.ic_media_play)
            return
        }

        mediaPlayer?.stop()
        mediaPlayer?.release()
        playingIcon?.setImageResource(android.R.drawable.ic_media_play)

        try {
            mediaPlayer = android.media.MediaPlayer().apply {
                setDataSource(path)
                prepare()
                start()
                setOnCompletionListener {
                    iconView.setImageResource(android.R.drawable.ic_media_play)
                    playingPath = null
                    playingIcon = null
                    it.release()
                }
            }
            playingPath = path
            playingIcon = iconView
            iconView.setImageResource(android.R.drawable.ic_media_pause)
        } catch (e: Exception) {
            e.printStackTrace()
            playingPath = null
            playingIcon = null
            iconView.setImageResource(android.R.drawable.ic_media_play)
        }
    }

    override fun getItemCount() = messages.size
}