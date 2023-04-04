using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;
using UnityEngine.Events;
using UnityEngine.Tilemaps;
using UnityEngine.Rendering;

public class HighlightProcessor : MonoBehaviour, IPointerDownHandler, IDragHandler, IPointerUpHandler
{

    public Texture2D originalImage;
   // public Texture2D editedImage;

    Vector2 imgSize;
    bool draw;
    Vector3[] corners;
    Vector2 min;
    Vector2 max;
    Vector2 size;

    Vector3 screenStartPos;
    Vector3 screenEndPos;

   public Vector2Int startPos;
    public Vector2Int endPos;

    int count = 0;

    HighlightData scene;

    // Start is called before the first frame update
    void Start()
    {
       
    }

    // Update is called once per frame
    void Update()
    {
        //if (count < 5)
       // {
        //    getInitSettings();
       //     count++;
       // }
    }

    public void OnPointerDown(PointerEventData eventData)
    {
        draw = true;
        Vector2 position = new Vector2(Input.mousePosition.x, Input.mousePosition.y) - min;

        screenStartPos = Input.mousePosition;

        Vector2 normalize = new Vector2(position.x / size.x, position.y / size.y);

        Vector2 pixelPos = normalize * imgSize;

        startPos = new Vector2Int(Mathf.FloorToInt(pixelPos.x), Mathf.FloorToInt(pixelPos.y));

      //  Debug.Log("Mouse: " + Input.mousePosition);
       // Debug.Log("StartPos: " +startPos);
        

    }
    public void OnDrag(PointerEventData eventData)
    {
        //Do I need the draw?
        if (draw)
        {

            Vector2 position = new Vector2(Input.mousePosition.x, Input.mousePosition.y) - min;

            screenEndPos = Input.mousePosition;

            Vector2 normalize = new Vector2(position.x / size.x, position.y / size.y);

            Vector2 pixelPos = normalize * imgSize;

            if (pixelPos.x >= 0 && pixelPos.x < imgSize.x)
            {
                if (pixelPos.y >= 0 && pixelPos.y < imgSize.y)
                {
                    endPos = new Vector2Int(Mathf.FloorToInt(pixelPos.x), Mathf.FloorToInt(pixelPos.y));

                    Vector3 worldStart = Camera.main.ScreenToWorldPoint(screenStartPos + Vector3.forward * 10);
                    Vector3 worldEnd = Camera.main.ScreenToWorldPoint(screenEndPos + Vector3.forward * 10);

                    DrawSquare(worldStart, worldEnd);
                }
            }

          

        }
    }

    public void OnPointerUp(PointerEventData eventData)
    {
        draw = false;
       // Debug.Log("EndPos: " + endPos);

        scene.givePos(startPos, endPos);
    }

    public void getInitSettings()
    {
       
        corners = new Vector3[4];
        this.GetComponent<RectTransform>().GetWorldCorners(corners);

        min = new Vector2(Camera.main.WorldToScreenPoint(corners[0]).x, Camera.main.WorldToScreenPoint(corners[0]).y);
        max = new Vector2(Camera.main.WorldToScreenPoint(corners[2]).x, Camera.main.WorldToScreenPoint(corners[2]).y);

        size = max - min;

        Debug.Log(min);
        Debug.Log(max);

        imgSize = new Vector2(originalImage.width, originalImage.height);
    }

    public void DrawLine(Vector3 start, Vector3 end, Color color, GameObject parent)
    {
        GameObject line = new GameObject("ThinLine");
        line.transform.localPosition = start;
        line.AddComponent<LineRenderer>();
        line.transform.parent = parent.transform;
        LineRenderer lr = line.GetComponent<LineRenderer>();
        lr.material.color = color;
        lr.startWidth = 0.1f;
        lr.endWidth = 0.1f;
        lr.SetPosition(0, start);
        lr.SetPosition(1, end);
    }

    public void DrawSquare(Vector3 start, Vector3 end)
    {

        destroyChildren(this.gameObject);


        //Corner 2                   Corner 3

        //Corner 1                   Corner 4
        Vector3 Corner1 = start;
        Vector3 Corner2 = new Vector3(start.x, end.y, start.z);
        Corner2.z = end.z;
        Vector3 Corner3 = end;
        Vector3 Corner4 = new Vector3(end.x, start.y, end.z);
        Corner4.z = Corner1.z;

        DrawLine(Corner1, Corner2, Color.green, this.gameObject);
        DrawLine(Corner2, Corner3, Color.green, this.gameObject);
        DrawLine(Corner3, Corner4, Color.green, this.gameObject);
        DrawLine(Corner4, Corner1, Color.green, this.gameObject);

        //Make a system here that checks for which vector is the lowest in both dimensions (equivalent to the start pos where the house will be) (Maybe make a function for that) 
    }

    public void destroyChildren(GameObject Obj)
    {
        //Only deletes children under the one referenced
        foreach (Transform child in Obj.transform)
        {
            //Safe to delete
            Destroy(child.gameObject);
        }
    }

    public void sendScene (HighlightData highlight)
    {
        scene = highlight;
    }






}
