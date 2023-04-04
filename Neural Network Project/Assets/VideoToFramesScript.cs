using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using FlexUI;
using UnityEngine.UI;
using UnityEngine.Video;
using System.IO;

public class VideoToFramesScript : MonoBehaviour
{

    [SerializeField] VideoClip vid;
    [SerializeField] VideoPlayer player;
    [SerializeField] string exportPath;

    [Header("UI")]
    [SerializeField] RectTransform holder;
    [SerializeField] RenderTexture vidText;

    int index = 0;

    // Start is called before the first frame update
    void Start()
    {
        //vidText.width = (int)vid.width;
        //vidText.height = (int)vid.height;

        setUI();

        setVideo();
    }

    // Update is called once per frame
    void Update()
    {
       
    }

    void setUI ()
    {
        Flex Holder = new Flex(holder, 1);

        Flex Image = new Flex(Holder.getChild(0), 10, Holder);
        Flex Save = new Flex(Holder.getChild(1), 1, Holder);

        Holder.setSize(new Vector2(Screen.width, Screen.height));
    }

    void setVideo ()
    {
        holder.GetChild(0).GetComponent<VideoPlayer>().targetTexture = vidText;


        player.Stop();
        player.renderMode = VideoRenderMode.APIOnly;
        player.prepareCompleted += Prepared;
        player.sendFrameReadyEvents = true;
        player.frameReady += FrameReady;
        player.Prepare();
        
        // holder.GetChild(0).GetComponent<VideoPlayer>().playOnAwake = true;

    }

    void Prepared(VideoPlayer vp) => vp.Pause();

    void FrameReady(VideoPlayer vp, long frameIndex)
    {
        //Debug.Log("FrameReady " + frameIndex);
        var textureToCopy = vp.texture;

        // Perform texture copy here ...
        vp.frame = frameIndex + 5;

        vidText = (RenderTexture)textureToCopy;

        byte[] bytes = toTexture2D(vidText).EncodeToPNG();
        File.WriteAllBytes(exportPath + "/" + "Frame" + " - " + index + ".png", bytes);
        Debug.Log("Saved Frame " + index);
        index++;


        //Graphics.CopyTexture(textureToCopy, dest);
        //Graphics.ConvertTexture(textureToCopy, 0, dest, 0);


    }

    Texture2D toTexture2D(RenderTexture rTex)
    {
        Texture2D tex = new Texture2D(rTex.width, rTex.height, TextureFormat.RGBA32, false);
        // ReadPixels looks at the active RenderTexture.
        RenderTexture.active = rTex;
        tex.ReadPixels(new Rect(0, 0, rTex.width, rTex.height), 0, 0);
        tex.Apply();
        return tex;
    }
}
