function set_item(tag, val){
    window.localStorage.setItem(tag, val);
}

function get_item(tag){
    return window.localStorage.getItem(tag);
}

export {set_item, get_item}; 