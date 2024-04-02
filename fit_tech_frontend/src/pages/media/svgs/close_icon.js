import React from "react";

function Close_Icon(props) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={props.height}
      height={props.height}
      viewBox="0 0 24 24"
      style = {props.style}
      onClick = {props.onClick}
    >
      <path d="M22.829 9.172L18.95 5.293a1 1 0 00-1.414 1.414l3.879 3.879a2.057 2.057 0 01.3.39c-.015 0-.027-.008-.042-.008L5.989 11a1 1 0 000 2l15.678-.032c.028 0 .051-.014.078-.016a2 2 0 01-.334.462l-3.879 3.879a1 1 0 101.414 1.414l3.879-3.879a4 4 0 000-5.656z"></path>
      <path d="M7 22H5a3 3 0 01-3-3V5a3 3 0 013-3h2a1 1 0 000-2H5a5.006 5.006 0 00-5 5v14a5.006 5.006 0 005 5h2a1 1 0 000-2z"></path>
    </svg>
  );
}

export default Close_Icon;
